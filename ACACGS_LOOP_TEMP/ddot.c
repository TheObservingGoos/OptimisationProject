#include "ddot.h"
#include <omp.h>

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
  omp_set_num_threads(4);
  double local_result = 0.0;
  
  int loopFactor = 4;
  int loopN = (n/loopFactor)*loopFactor;
  int i;  

  if (y==x){
    #pragma omp parallel for lastprivate(i) firstprivate(loopFactor, loopN) reduction(+:local_result) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      local_result += x[i]*x[i];
      local_result += x[i+1]*x[i+1];
      local_result += x[i+2]*x[i+2];
      local_result += x[i+3]*x[i+3];
    }
    for (; i<n; i++){
      local_result += x[i]*x[i];
    }
  } else {
    #pragma omp parallel for lastprivate(i) firstprivate(loopFactor, loopN) reduction(+:local_result) schedule(guided)
    for (i=0; i<loopN; i+=loopFactor) {
      local_result += x[i]*y[i];
      local_result += x[i+1]*y[i+1];
      local_result += x[i+2]*y[i+2];
      local_result += x[i+3]*y[i+3];
    }
    for (; i<n; i++){
      local_result += x[i]*y[i];
    }
  }
  *result = local_result;

  return 0;
}
