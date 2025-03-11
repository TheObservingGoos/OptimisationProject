#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const float * const x, float * const y)
{
    const int nrow = (const int) A->local_nrow;
    int loopFactor = 4;
    int j;  
    
    #pragma omp parallel for lastprivate(j) firstprivate(loopFactor) schedule(guided)
    for (int i=0; i< nrow; i++) {
        float sum = 0.0;
        const float * const cur_vals = (const float * const) A->ptr_to_vals_in_row[i];
        const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
        const int cur_nnz = (const int) A->nnz_in_row[i];

        int loopN = (cur_nnz / loopFactor) * loopFactor;
        __m128 sumVec = _mm_setzero_ps(); // Initialize sum vector

        
        for (j=0; j<loopN; j+=loopFactor) {
            __m128 valsVec = _mm_loadu_ps(&cur_vals[j]);
            __m128 xVec = _mm_set_ps(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]);
            
            sumVec = _mm_add_ps(sumVec, _mm_mul_ps(valsVec, xVec));
        }

        // Horizontal addition of sumVec elements
        float temp[4];
        _mm_storeu_ps(temp, sumVec);
        sum = temp[0] + temp[1] + temp[2] + temp[3];

        for (; j<cur_nnz; j++){
            sum += cur_vals[j]*x[cur_inds[j]];
        }
        y[i] = sum;
    }
    return 0;
}