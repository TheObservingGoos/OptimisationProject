#include "waxpby.h"

#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <xmmintrin.h>
#else
	#include <GL/glut.h>
	#include <immintrin.h>
#endif

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {

  int loopFactor = 4;
  int i;

  int loopN = (n/loopFactor)*loopFactor;

  __m256d betaVec = _mm256_set1_pd(beta);
  __m256d alphaVec = _mm256_set1_pd(alpha);

  if (alpha==1.0) {
    for (i=0; i<loopN; i+=loopFactor) {
      __m256d wVec = _mm256_loadu_pd(w+i);
      __m256d xVec = _mm256_loadu_pd(x+i);
      __m256d yVec = _mm256_loadu_pd(y+i);

      __m256d betaTotalVec = _mm256_mul_pd(betaVec, yVec);
      wVec = _mm256_add_pd(betaTotalVec, xVec);
      
      _mm256_storeu_pd(w+i, wVec);
    }
    for(; i<n; i++){
      w[i] = x[i] + beta * y[i];
    }
  } else if(beta==1.0) {
    for (i=0; i<loopN; i+=loopFactor) {
      __m256d wVec = _mm256_loadu_pd(w+i);
      __m256d xVec = _mm256_loadu_pd(x+i);
      __m256d yVec = _mm256_loadu_pd(y+i);

      __m256d alphaTotalVec = _mm256_mul_pd(alphaVec, xVec);
      wVec = _mm256_add_pd(alphaTotalVec, yVec);

      _mm256_storeu_pd(w+i, wVec);
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + y[i];
    }
  } else {
    for (i=0; i<loopN; i+=loopFactor) {

      __m256d wVec = _mm256_loadu_pd(w+i);
      __m256d xVec = _mm256_loadu_pd(x+i);
      __m256d yVec = _mm256_loadu_pd(y+i);

      __m256d alphaTotalVec = _mm256_mul_pd(alphaVec, xVec);
      __m256d betaTotalVec = _mm256_mul_pd(betaVec, yVec);
      wVec = _mm256_add_pd(alphaTotalVec, betaTotalVec);

      _mm256_storeu_pd(w+i, wVec);
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}
