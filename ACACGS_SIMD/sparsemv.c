#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>

#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y)
{
    const int nrow = (const int) A->local_nrow;
    int loopFactor = 4;
    int j;  

    for (int i=0; i< nrow; i++) {
        double sum = 0.0;
        const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
        const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
        const int cur_nnz = (const int) A->nnz_in_row[i];

        int loopN = (cur_nnz / loopFactor) * loopFactor;
        __m256d sumVec = _mm256_setzero_pd(); // Initialize sum vector

        for (j=0; j<loopN; j+=loopFactor) {
            __m256d valsVec = _mm256_loadu_pd(&cur_vals[j]);
            __m256d xVec = _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]);
            
            sumVec = _mm256_add_pd(sumVec, _mm256_mul_pd(valsVec, xVec));
        }

        // Horizontal addition of sumVec elements
        double temp[4];
        _mm256_storeu_pd(temp, sumVec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        for (; j<cur_nnz; j++){
            sum += cur_vals[j]*x[cur_inds[j]];
        }
        y[i] = sum;
        }
    return 0;
}