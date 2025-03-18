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
int sparsemv(struct mesh *A, const double * const x, double * const y)
{
    omp_set_num_threads(4); // Set number of threads to be used to be 4

    // Divide the number of iterations to loop over by 4 to provide for each thread, as well as accomodating for double vectors (which will hold 4 at a time)
    const int nrow = (const int) A->local_nrow;
    int loopFactor = 4;
    int j;  
    
    // Allow the following outer (and inner) loop to run in parallel across all threads in a guided schedule
    #pragma omp parallel for lastprivate(j) firstprivate(loopFactor) schedule(guided)
    for (int i=0; i< nrow; i++) {
        double sum = 0.0;
        const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
        const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
        const int cur_nnz = (const int) A->nnz_in_row[i];

        int loopN = (cur_nnz / loopFactor) * loopFactor;
        __m256d sumVec = _mm256_setzero_pd(); // Initialize sum vector

        // Use vectorisation to reduce the number of instructions run to add all calculated values to temporary vectors
        for (j=0; j<loopN; j+=loopFactor) {
            // Load 4 values from cur_vals into a vector to multiply with their respective x values in a different vector
            __m256d valsVec = _mm256_loadu_pd(&cur_vals[j]); 
            __m256d xVec = _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]); // Remember that these vectors can hold 4 doubles each
            
            sumVec = _mm256_add_pd(sumVec, _mm256_mul_pd(valsVec, xVec)); // Do the vector calculation of sumVec, valsVec and xVec
        }

        // Providde horizontal addition of sumVec elements
        double temp[4]; // Make a temporary double array to store the summed values to
        _mm256_storeu_pd(temp, sumVec);
        sum = temp[0] + temp[1] + temp[2] + temp[3]; // Add each vector value to a final sum

        for (; j<cur_nnz; j++){
            sum += cur_vals[j]*x[cur_inds[j]]; // Run the computation for any leftover variables not accessed during loop unrolling
        }
        y[i] = sum; 
    }
    return 0;
}