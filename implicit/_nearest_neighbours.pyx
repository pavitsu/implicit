import threading
import time
import warnings

import cython
import numpy as np
import scipy.sparse

from cython cimport floating, integral

from cython.operator import dereference
from cython.parallel import parallel, prange

from libcpp cimport bool
from libcpp.algorithm cimport sort_heap
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from tqdm.auto import tqdm

from implicit.utils import check_csr


cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass greater[T=*]:
        greater() except +


cdef extern from "implicit/nearest_neighbours.h" namespace "implicit" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results
        greater[pair[Value, Index]] heap_order

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index item_count)
        void add(Index index, Value value)
        void foreach[Function](Function & f)
        vector[Value] sums


cdef class NearestNeighboursScorer(object):
    """ Class to return the top K items from multiplying a users likes
    by a precomputed sparse similarity matrix. """
    cdef SparseMatrixMultiplier[int, double] * neighbours

    cdef int[:] similarity_indptr
    cdef int[:] similarity_indices
    cdef double[:] similarity_data

    cdef object lock

    def __cinit__(self, similarity):
        self.neighbours = new SparseMatrixMultiplier[int, double](similarity.shape[0])
        self.similarity_indptr = similarity.indptr
        self.similarity_indices = similarity.indices
        self.similarity_data = similarity.data.astype(np.float64)
        self.lock = threading.RLock()

    @cython.boundscheck(False)
    def recommend(self, int[:] user_indptr, int[:] user_indices, floating[:] user_data,
                  int K=10, bool remove_own_likes=True):
        cdef int index1, index2, i, count
        cdef double weight
        cdef double temp
        cdef pair[double, int] result

        cdef int[:] indices
        cdef double[:] data

        cdef TopK[int, double] * topK = new TopK[int, double](K)
        try:
            with self.lock:
                with nogil:
                    for index1 in range(user_indptr[0], user_indptr[1]):
                        i = user_indices[index1]
                        weight = user_data[index1]

                        for index2 in range(self.similarity_indptr[i], self.similarity_indptr[i+1]):
                            self.neighbours.add(self.similarity_indices[index2],
                                                self.similarity_data[index2] * weight)

                    if remove_own_likes:
                        # set the score to 0 for things already liked
                        for index1 in range(user_indptr[0], user_indptr[1]):
                            i = user_indices[index1]
                            self.neighbours.sums[i] = 0

                    self.neighbours.foreach(dereference(topK))

            count = topK.results.size()
            indices = ret_indices = np.zeros(count, dtype=np.int32)
            data = ret_data = np.zeros(count)

            sort_heap(topK.results.begin(), topK.results.end(), topK.heap_order)
            with nogil:
                i = 0
                for result in topK.results:
                    indices[i] = result.second
                    data[i] = result.first
                    i += 1
            return ret_indices, ret_data

        finally:
            del topK

    def __dealloc__(self):
        del self.neighbours


@cython.boundscheck(False)
def all_pairs_knn(users, unsigned int K=100, int num_threads=0, show_progress=True, mininterval = 0.1):
    """ Returns the top K nearest neighbours for each row in the matrix.
    """
    users = check_csr(users)
    items = users.T.tocsr()

    cdef int item_count = items.shape[0]
    cdef int i, u, index1, index2, j
    cdef double w1, w2

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    cdef double[:] item_data = items.data

    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    cdef double[:] user_data = users.data

    cdef SparseMatrixMultiplier[int, double] * neighbours
    cdef TopK[int, double] * topk
    cdef pair[double, int] result

    # holds triples of output
    cdef double[:] values = np.zeros(item_count * K)
    cdef long[:] rows = np.zeros(item_count * K, dtype=int)
    cdef long[:] cols = np.zeros(item_count * K, dtype=int)

    progress = tqdm(total=item_count, mininterval = mininterval, disable=not show_progress)
    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, double](item_count)
        topk = new TopK[int, double](K)

        try:
            for i in prange(item_count, schedule='dynamic', chunksize=8):
                for index1 in range(item_indptr[i], item_indptr[i+1]):
                    u = item_indices[index1]
                    w1 = item_data[index1]

                    for index2 in range(user_indptr[u], user_indptr[u+1]):
                        neighbours.add(user_indices[index2], user_data[index2] * w1)

                topk.results.clear()
                neighbours.foreach(dereference(topk))

                index2 = K * i

                for result in topk.results:
                    rows[index2] = i
                    cols[index2] = result.second
                    values[index2] = result.first
                    index2 = index2 + 1
                with gil:
                    progress.update(1)

        finally:
            del neighbours
            del topk
    progress.close()
    return scipy.sparse.coo_matrix((values, (rows, cols)),
                                   shape=(item_count, item_count))


@cython.boundscheck(False)
@cython.wraparound(False)
def conditional_probability_similarity(users, double alpha=0.5, bint row_normalize=False,
                                        unsigned int K=100, int num_threads=0, show_progress=True,
                                        mininterval = 0.1):
    """Computes the Conditional Probability-Based Item Similarity in parallel.
    
    Only computes top-K similar items per item to avoid memory issues.
    
    Parameters
    ----------
    users : csr_matrix
        User-Item matrix of shape (n_users, m_items)
    alpha : double
        Scaling parameter to penalize frequently purchased items [0, 1]
    row_normalize : bool
        If True, applies Formula 3 (gives more weight to smaller baskets)
    K : int
        Number of top similar items to keep per item
    num_threads : int
        Number of threads for parallel computation
    show_progress : bool
        
    Returns
    -------
    scipy.sparse.csr_matrix
        (m_items, m_items) similarity matrix with only K non-zero elements per row
    """
    users = check_csr(users)
    users = users.astype(np.float64)
    items = users.T.tocsr()

    cdef int m_items = items.shape[0]
    cdef int n_users = users.shape[0]
    cdef int i, u, index1, index2, j

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    cdef double[:] item_data = items.data

    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    cdef double[:] user_data = users.data

    item_freqs = np.array(users.sum(axis=0), dtype=np.float64).flatten()
    cdef double[:] item_freqs_view = item_freqs
    cdef double[:] freq_u_alpha = np.power(item_freqs, alpha)
    cdef double[:] freq_u_alpha_view = freq_u_alpha

    cdef SparseMatrixMultiplier[int, double] * neighbours
    cdef TopK[int, double] * topk
    cdef pair[double, int] result

    cdef double basket_size, row_norm, w1, w2, denom, numer
    cdef int result_size

    progress = tqdm(total=m_items, disable=not show_progress, mininterval = mininterval)

    cdef double[:] values = np.zeros(m_items * K)
    cdef long[:] rows = np.zeros(m_items * K, dtype=np.intp)
    cdef long[:] cols = np.zeros(m_items * K, dtype=np.intp)

    with nogil, parallel(num_threads=num_threads):
        neighbours = new SparseMatrixMultiplier[int, double](m_items)
        topk = new TopK[int, double](K)

        try:
            for i in prange(m_items, schedule='dynamic', chunksize=1):
                for index1 in range(item_indptr[i], item_indptr[i+1]):
                    u = item_indices[index1]
                    w1 = item_data[index1]

                    if row_normalize:
                        basket_size = 0
                        for index2 in range(user_indptr[u], user_indptr[u+1]):
                            basket_size = basket_size + 1.0
                        if basket_size > 0:
                            row_norm = basket_size ** 0.5
                        else:
                            row_norm = 1.0
                        w1 = w1 / row_norm

                    for index2 in range(user_indptr[u], user_indptr[u+1]):
                        w2 = user_data[index2]
                        neighbours.add(user_indices[index2], w2 * w1)

                topk.results.clear()
                neighbours.foreach(dereference(topk))

                index2 = K * i

                for result in topk.results:
                    rows[index2] = i
                    cols[index2] = result.second
                    numer = result.first
                    with gil:
                        denom = item_freqs_view[i] * (freq_u_alpha_view[result.second])
                        if denom > 0:
                            values[index2] = numer / denom
                        else:
                            values[index2] = 0.0
                    index2 = index2 + 1

                with gil:
                    progress.update(1)

        finally:
            del neighbours
            del topk

    progress.close()

    return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(m_items, m_items))
