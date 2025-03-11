#include "waxpby.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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
int waxpby (const int n, const float alpha, const float * const x, const float beta, const float * const y, float * const w) {

  int loopFactor = 4;
  int i = 0;

  int loopN = (n/loopFactor)*loopFactor;

  __m128 betaVec = _mm_set1_ps(beta);
  __m128 alphaVec = _mm_set1_ps(alpha);

  // printf("alpha: %e\n\n", alpha);
  // printf("beta: %e\n\n", beta);

  if (alpha==1.0) {
    #pragma omp parallel for firstprivate(loopFactor, loopN) lastprivate(i) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      // printf("Number of Threads: %d\n\n", omp_get_num_threads());
      // printf("Thread: %d\n\n", omp_get_thread_num());
      __m128 xVec = _mm_loadu_ps(x+i);
      __m128 yVec = _mm_loadu_ps(y+i);

      __m128 betaTotalVec = _mm_mul_ps(betaVec, yVec);
      __m128 wVec = _mm_add_ps(betaTotalVec, xVec);
      
      _mm_storeu_ps(w+i, wVec);
    }
    // printf("n: %d\ni: %d\n\n", n, i);
    for(; i<n; i++){
      w[i] = x[i] + beta * y[i];
    }

  } else if(beta==1.0) {
    for (i=0; i<loopN; i+=loopFactor) {
      __m128 xVec = _mm_loadu_ps(x+i);
      __m128 yVec = _mm_loadu_ps(y+i);

      __m128 alphaTotalVec = _mm_mul_ps(alphaVec, xVec);
      __m128 wVec = _mm_add_ps(alphaTotalVec, yVec);

      _mm_storeu_ps(w+i, wVec);
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + y[i];
    }

  } else {
    for (i=0; i<loopN; i+=loopFactor) {
      __m128 xVec = _mm_loadu_ps(x+i);
      __m128 yVec = _mm_loadu_ps(y+i);

      __m128 alphaTotalVec = _mm_mul_ps(alphaVec, xVec);
      __m128 betaTotalVec = _mm_mul_ps(betaVec, yVec);
      __m128 wVec = _mm_add_ps(alphaTotalVec, betaTotalVec);

      _mm_storeu_ps(w+i, wVec);
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + beta * y[i];
    }
    
  }

  return 0;
}