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
  double local_result = 0.0;
  
  int loopFactor = 4;
  int loopN = (n/loopFactor)*loopFactor;
  int i;  

  if (y==x){
    for (i=0; i<loopN; i+=loopFactor) {
      __m256d xVec = _mm256_loadu_pd(x+i);
      __m256d sumVec = _mm256_setzero_pd();
      xVec = _mm256_mul_pd(xVec, xVec);
      sumVec = _mm256_add_pd(sumVec, xVec);
      double temp[4];
      _mm256_storeu_pd(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3];
    }
    for (; i<n; i++){
      local_result += x[i]*x[i];
    }
  } else {
    for (i=0; i<loopN; i+=loopFactor) {
      __m256d xVec = _mm256_loadu_pd(x+i);
      __m256d yVec = _mm256_loadu_pd(y+i);
      __m256d sumVec = _mm256_setzero_pd();
      xVec = _mm256_mul_pd(xVec, yVec);
      sumVec = _mm256_add_pd(sumVec, xVec);
      double temp[4];
      _mm256_storeu_pd(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3];
    }
    for (; i<n; i++) {
      local_result += x[i]*y[i];
    }
  }
  *result = local_result;

  return 0;
}
