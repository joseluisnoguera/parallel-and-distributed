#include <stdio.h>
#include "omp.h"

void main() {
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    printf("Hello from thread %d\n", id);
  }
}
