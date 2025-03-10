#include "ddot.h"

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
  
  int loopFactor = 8;
  int loopN = (n/loopFactor)*loopFactor;
  int i;  

  if (y==x){
    for (i=0; i<loopN; i+=loopFactor) {
      local_result += x[i]*x[i];
      local_result += x[i+1]*x[i+1];
      local_result += x[i+2]*x[i+2];
      local_result += x[i+3]*x[i+3];
      local_result += x[i+4]*x[i+4];
      local_result += x[i+5]*x[i+5];
      local_result += x[i+6]*x[i+6];
      local_result += x[i+7]*x[i+7];
    }
    for (; i<n; i++){
      local_result += x[i]*x[i];
    }
  } else {
    for (i=0; i<loopN; i+=loopFactor) {
      local_result += x[i]*y[i];
      local_result += x[i+1]*y[i+1];
      local_result += x[i+2]*y[i+2];
      local_result += x[i+3]*y[i+3];
      local_result += x[i+4]*y[i+4];
      local_result += x[i+5]*y[i+5];
      local_result += x[i+6]*y[i+6];
      local_result += x[i+7]*y[i+7];
    }
    for (; i<n; i++){
      local_result += x[i]*y[i];
    }
  }
  *result = local_result;

  return 0;
}
