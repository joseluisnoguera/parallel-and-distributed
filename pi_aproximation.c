#include <stdio.h>
#include "omp.h"
#include <time.h>

// Aproximation to number Pi by the Gregory-Leibniz series

static long runs = 500;
static long num_steps = 100000;
double step;

void aproximate_pi_a() {
  int i;
  double x, pi, sum = 0.0;

  step = 1.0 / (double) num_steps;

  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

void aproximate_pi_b() {
  int i;
  double x, pi, sum = 0.0;

  step = 1.0 / (double) num_steps;

  #pragma omp parallel for private(x)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    #pragma omp atomic
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

void aproximate_pi_c() {
  int i;
  double x, pi, sum = 0.0;

  step = 1.0 / (double) num_steps;

  #pragma omp parallel for private(x) reduction(+:sum)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

double calculate_cpu_utilization(void (*func)(void)) {
  clock_t start, end;
  double cpu_time_used_accum = 0.0;

  for (int i = 0; i < runs; i++) {
    start = clock();
    func();
    end = clock();
    cpu_time_used_accum += (double) (end - start) / CLOCKS_PER_SEC;
  }

  double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  return cpu_time_used;
}

void main() {

  double cpu_time_used_a = calculate_cpu_utilization(aproximate_pi_a);
  double cpu_time_used_b = calculate_cpu_utilization(aproximate_pi_b);
  double cpu_time_used_c = calculate_cpu_utilization(aproximate_pi_c);

  printf("Without omp: %.6f\n", cpu_time_used_a);
  printf("Using parallel for: %.6f\n", cpu_time_used_b);
  printf("Using parallel for and reduce: %.6f\n", cpu_time_used_c);


}