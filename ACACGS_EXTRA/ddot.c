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
int ddot (const int n, const float * const x, const float * const y, float * const result) {  
  float local_result = 0.0;
  
  int loopFactor = 4;
  int loopN = (n/loopFactor)*loopFactor;
  int i;  

  if (y==x){
    for (i=0; i<loopN; i+=loopFactor) {
      __m128 xVec = _mm_loadu_ps(x+i);
      __m128 sumVec = _mm_setzero_ps();
      xVec = _mm_mul_ps(xVec, xVec);
      sumVec = _mm_add_ps(sumVec, xVec);
      float temp[4];
      _mm_storeu_ps(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3];
    }
    for (; i<n; i++){
      local_result += x[i]*x[i];
    }
  } else {
    for (i=0; i<loopN; i+=loopFactor) {
      __m128 xVec = _mm_loadu_ps(x+i);
      __m128 yVec = _mm_loadu_ps(y+i);
      __m128 sumVec = _mm_setzero_ps();
      xVec = _mm_mul_ps(xVec, yVec);
      sumVec = _mm_add_ps(sumVec, xVec);
      float temp[4];
      _mm_storeu_ps(temp, sumVec);
      local_result += temp[0] + temp[1] + temp[2] + temp[3];
    }
    for (; i<n; i++) {
      local_result += x[i]*y[i];
    }
  }
  *result = local_result;

  return 0;
}
