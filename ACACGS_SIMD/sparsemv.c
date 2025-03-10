#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#include "sparsemv.h"

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <xmmintrin.h>
#else
	#include <GL/glut.h>
	#include <immintrin.h>
#endif

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y) {
    const int nrow = A->local_nrow;
    const int loopFactor = 4;  // AVX works with 4 doubles (256-bit)

    for (int i = 0; i < nrow; i++) {
        double sum = 0.0;
        const double *cur_vals = A->ptr_to_vals_in_row[i];
        const int *cur_inds = A->ptr_to_inds_in_row[i];
        const int cur_nnz = A->nnz_in_row[i];

        __m256d sum_vec = _mm256_setzero_pd();  // Vectorized sum
        int j;

        // Process 4 elements at a time (AVX)
        int loopN = (cur_nnz / loopFactor) * loopFactor;
        for (j = 0; j < loopN; j += loopFactor) {
            // Load 4 values from cur_vals
            __m256d val_vec = _mm256_loadu_pd(&cur_vals[j]);

            // Load 4 indices and gather values from x
            __m256d x_vec = _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], 
                                          x[cur_inds[j+1]], x[cur_inds[j]]);

            // Multiply element-wise
            __m256d prod = _mm256_mul_pd(val_vec, x_vec);

            // Accumulate into sum_vec
            sum_vec = _mm256_add_pd(sum_vec, prod);
        }

        // Reduce the sum_vec into a single sum
        __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);  // Upper half
        __m128d sum_low  = _mm256_castpd256_pd128(sum_vec);    // Lower half
        __m128d sum_128  = _mm_add_pd(sum_low, sum_high);  // Add high and low
        double sum_arr[2];
        _mm_storeu_pd(sum_arr, sum_128);
        sum += sum_arr[0] + sum_arr[1];

        // Handle remaining elements (scalar fallback)
        for (; j < cur_nnz; j++) {
            sum += cur_vals[j] * x[cur_inds[j]];
        }

        y[i] = sum;
    }

    return 0;
}
