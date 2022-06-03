#include <stdio.h>
// #include <math.h>
#include <time.h>
#include <stdarg.h>
#include "pi_approximation.c"
#include "matrix_multiplication.c"

#define RUNS 50

double mean(double *array, int size) {
  int i;
  double sum = 0.0;
  for (i=0; i < size; i++) {
    sum += array[i];
  }
  return sum / size;
}

// double standard_deviation(double *array, int size, double mean) {
//   int i;
//   double sum = 0.0;
//   for (i=0; i < size; i++) {
//     sum += (array[i] - mean) * (array[i] - mean);
//   }
//   return sqrt(sum / size);
// }

double calculate_cpu_utilization_pi_approx(void (*func)(int, long), int threads, long num_steps) {
  clock_t start, end;
  double cpu_times_used[RUNS];
  double cpu_time_mean = 0.0;

  for (int i = 0; i < RUNS; i++) {
    start = clock();
    func(threads, num_steps);
    end = clock();
    cpu_times_used[i] = (double) (end - start) / CLOCKS_PER_SEC;
  }

  cpu_time_mean = mean(cpu_times_used, RUNS);

  //* Used for decide the number of RUNS
  // double cpu_time_standard_deviation = standard_deviation(cpu_times_used, RUNS, cpu_time_mean);
  // printf("Standard deviation: %.6f\n", cpu_time_standard_deviation * 1000);

  return cpu_time_mean;
}

double calculate_cpu_utilization_matrix_mult(void (*func)(int, struct Matrix*, struct Matrix*, struct Matrix*), int threads, struct Matrix *A, struct Matrix *B, struct Matrix *C) {
  clock_t start, end;
  double cpu_times_used[RUNS];
  double cpu_time_mean = 0.0;

  for (int i = 0; i < RUNS; i++) {
    start = clock();
    func(threads, A, B , C);
    end = clock();
    cpu_times_used[i] = (double) (end - start) / CLOCKS_PER_SEC;
  }

  cpu_time_mean = mean(cpu_times_used, RUNS);

  //* Used for decide the number of RUNS
  // double cpu_time_standard_deviation = standard_deviation(cpu_times_used, RUNS, cpu_time_mean);
  // printf("Standard deviation: %.6f\n", cpu_time_standard_deviation * 1000);

  return cpu_time_mean;
}

void main() {
  FILE *fp_pi_ap;
  FILE *fp_m_m;
  fp_pi_ap = fopen("pi_approximation.csv", "w");
  fp_m_m = fopen("matrix_multiplication.csv", "w");

  fprintf(fp_pi_ap, "threads,num_steps,pi_approximation_without_omp,pi_approximation_parallel_for,pi_approximation_parallel_for_reduction");
  fprintf(fp_m_m, "threads,num_steps,matrix_multiplication__without_omp, matrix_multiplication_parallel_for, matrix_multiplication_parallel_for_reduction");

  for (int threads = 2; threads < 9; threads = threads + 2) {
    // Pi approximation
    printf("\nStatistics for pi approximation:\n");

    long num_steps_array[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
    int pi_approximation_test_runs = sizeof(num_steps_array) / sizeof(num_steps_array[0]);

    for (int i = 0; i < pi_approximation_test_runs; i++) {
      long num_steps = num_steps_array[i];

      double stats_without_omp = calculate_cpu_utilization_pi_approx(aproximate_pi_without_omp, threads, num_steps);
      double stats_parallel_for = calculate_cpu_utilization_pi_approx(aproximate_pi_parallel_for, threads, num_steps);
      double stats_parallel_for_and_reduction = calculate_cpu_utilization_pi_approx(aproximate_pi_parallel_for_reduction, threads, num_steps);

      printf("Threads: %d Number of steps:%ld\n", threads, num_steps);
      fprintf(fp_pi_ap, "\n%d,%ld,%.6f,%.6f,%.6f", threads, num_steps, stats_without_omp, stats_parallel_for, stats_parallel_for_and_reduction);
    }

    // Matrix multiplication
    printf("\nStatistics for matrix multiplication:\n");

    // Square matrices for simplicity
    int matrix_dimensions[] = {10, 50, 100, 250};
    int matrix_multiplication_test_runs = sizeof(matrix_dimensions) / sizeof(matrix_dimensions[0]);

    for(int i = 0; i < matrix_multiplication_test_runs; i++) {
      int matrix_dimension = matrix_dimensions[i];

      struct Matrix* A = create_random_matrix(matrix_dimension, matrix_dimension);
      struct Matrix* B = create_random_matrix(matrix_dimension, matrix_dimension);
      struct Matrix* C = create_random_matrix(matrix_dimension, matrix_dimension);

      double stats_matrix_multiplication_without_omp = calculate_cpu_utilization_matrix_mult(matrix_multiplication_without_omp, threads, A, B, C);
      double stats_matrix_multiplication_parallel_for = calculate_cpu_utilization_matrix_mult(matrix_multiplication_parallel_for, threads, A, B, C);
      double stats_matrix_multiplication_parallel_for_reduce = calculate_cpu_utilization_matrix_mult(matrix_multiplication_parallel_for_reduce, threads, A, B, C);

      printf("Threads: %d Matrix dimension:%d\n", threads, matrix_dimension);
      fprintf(fp_m_m, "\n%d,%d,%.6f,%.6f,%.6f", threads, matrix_dimension, stats_matrix_multiplication_without_omp, stats_matrix_multiplication_parallel_for, stats_matrix_multiplication_parallel_for_reduce);
    }
  }


}