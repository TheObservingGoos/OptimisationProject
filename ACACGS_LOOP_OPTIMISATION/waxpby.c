#include "waxpby.h"

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

  int loopFactor = 8;
  int i;

  int loopN = (n/loopFactor)*loopFactor;

  if (alpha==1.0) {
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = x[i] + beta * y[i];
      w[i+1] = x[i+1] + beta * y[i+1];
      w[i+2] = x[i+2] + beta * y[i+2];
      w[i+3] = x[i+3] + beta * y[i+3];
      w[i+4] = x[i+4] + beta * y[i+4];
      w[i+5] = x[i+5] + beta * y[i+5];
      w[i+6] = x[i+6] + beta * y[i+6];
      w[i+7] = x[i+7] + beta * y[i+7];

    }
    for(; i<n; i++){
      w[i] = x[i] + beta * y[i];
    }
  } else if(beta==1.0) {
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + y[i];
      w[i+1] = alpha * x[i+1] + y[i+1];
      w[i+2] = alpha * x[i+2] + y[i+2];
      w[i+3] = alpha * x[i+3] + y[i+3];
      w[i+4] = alpha * x[i+4] + y[i+4];
      w[i+5] = alpha * x[i+5] + y[i+5];
      w[i+6] = alpha * x[i+6] + y[i+6];
      w[i+7] = alpha * x[i+7] + y[i+7];
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + y[i];
    }
  } else {
    for (i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + beta * y[i];
      w[i+1] = alpha * x[i+1] + beta * y[i+1];
      w[i+2] = alpha * x[i+2] + beta * y[i+2];
      w[i+3] = alpha * x[i+3] + beta * y[i+3];
      w[i+4] = alpha * x[i+4] + beta * y[i+4];
      w[i+5] = alpha * x[i+5] + beta * y[i+5];
      w[i+6] = alpha * x[i+6] + beta * y[i+6];
      w[i+7] = alpha * x[i+7] + beta * y[i+7];
    }
    for(; i<n; i++){
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}
