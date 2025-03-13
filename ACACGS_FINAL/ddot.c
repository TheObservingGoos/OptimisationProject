#include "ddot.h"
#include <omp.h>

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <xmmintrin.h>
#else
	#include <GL/glut.h>
	#include <immintrin.h>
#endif

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  omp_set_num_threads(4); // Set number of threads to be used to be 4

  double local_result = 0.0;
  
  // Divide the number of iterations to loop over by 4 to provide for each thread, as well as accomodating for the double vectors (which will hold 4 
  // doubles at a time)
  int loopFactor = 4;
  int loopN = (n/loopFactor)*loopFactor; // Ensure the loopN is divisible by 4
  int i;
  double temp[4]; // Create a temporary double array to store 

  // Use vectorisation to reduce the number of instructions run to add all calculated values to temporary vectors
  if (y==x){
    #pragma omp parallel for private(temp) lastprivate(i) firstprivate(loopFactor, loopN) reduction(+:local_result) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {

      __m256d xVec = _mm256_loadu_pd(x+i); // Load 4 values from x into a vector to multiply with itself
      __m256d sumVec = _mm256_setzero_pd(); // Initialize sum vector

      xVec = _mm256_mul_pd(xVec, xVec);
      sumVec = _mm256_add_pd(sumVec, xVec); // Add the x vector to the current sum vector in an incremental format

      _mm256_storeu_pd(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3]; // Add the temp vector array to local_result, using OpenMP to handle potential race conditions
    }
    for (; i<n; i++){
      local_result += x[i]*x[i]; // Run the computation for any leftover variables not accessed during loop unrolling
    }
  } else {
    // Allow the following loop to run in parallel across all threads in a guided schedule
    // local_result is set in reduction as it is a shared variable to be added to
    #pragma omp parallel for private(temp) lastprivate(i) firstprivate(loopFactor, loopN) reduction(+:local_result) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {

      __m256d xVec = _mm256_loadu_pd(x+i); // Load 4 values from x into a vector to multiply with another 4 values for y, also in its own vector
      __m256d yVec = _mm256_loadu_pd(y+i);
      
      __m256d sumVec = _mm256_setzero_pd();
      xVec = _mm256_mul_pd(xVec, yVec);
      sumVec = _mm256_add_pd(sumVec, xVec); // Add the x vector to the current sum vector in an incremental format
      
      _mm256_storeu_pd(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3]; // Add the temp vector array to local_result, using OpenMP to handle potential race conditions
    }
    for (; i<n; i++) {
      local_result += x[i]*y[i]; // Run the computation for any leftover variables not accessed during loop unrolling
    }
  }
  *result = local_result;

  return 0;
}
