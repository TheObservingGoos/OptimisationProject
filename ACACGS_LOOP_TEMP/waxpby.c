#include "waxpby.h"
#include <omp.h>

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
  omp_set_num_threads(4);

  int loopFactor = 4;
  int i;

  int loopN = (n/loopFactor)*loopFactor;

  if (alpha==1.0) {
    #pragma omp parallel for firstprivate(loopFactor, loopN, beta) lastprivate(i) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = x[i] + beta * y[i];
      w[i+1] = x[i+1] + beta * y[i+1];
      w[i+2] = x[i+2] + beta * y[i+2];
      w[i+3] = x[i+3] + beta * y[i+3];

    }
    for(; i<n; i++){
      w[i] = x[i] + beta * y[i];
    }
  } else if(beta==1.0) {
    #pragma omp parallel for firstprivate(loopFactor, loopN, alpha) lastprivate(i) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + y[i];
      w[i+1] = alpha * x[i+1] + y[i+1];
      w[i+2] = alpha * x[i+2] + y[i+2];
      w[i+3] = alpha * x[i+3] + y[i+3];
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + y[i];
    }
  } else {
    #pragma omp parallel for firstprivate(loopFactor, loopN, alpha, beta) lastprivate(i) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + beta * y[i];
      w[i+1] = alpha * x[i+1] + beta * y[i+1];
      w[i+2] = alpha * x[i+2] + beta * y[i+2];
      w[i+3] = alpha * x[i+3] + beta * y[i+3];
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}
