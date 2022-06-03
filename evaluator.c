#include <stdio.h>
// #include <math.h>
#include <time.h>
#include <stdarg.h>
#include "pi_approximation.c"
#include "matrix_multiplication.c"

#define RUNS 30

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

double calculate_cpu_utilization_pi_approx(void (*func)(int, long, int), int threads, long num_steps, int chunk_size) {
  clock_t start, end;
  double cpu_times_used[RUNS];
  double cpu_time_mean = 0.0;

  for (int i = 0; i < RUNS; i++) {
    start = clock();
    func(threads, num_steps, chunk_size);
    end = clock();
    cpu_times_used[i] = (double) (end - start) / CLOCKS_PER_SEC;
  }

  cpu_time_mean = mean(cpu_times_used, RUNS);

  //* Used for decide the number of RUNS
  // double cpu_time_standard_deviation = standard_deviation(cpu_times_used, RUNS, cpu_time_mean);
  // printf("Standard deviation: %.6f\n", cpu_time_standard_deviation * 1000);

  return cpu_time_mean;
}

double calculate_cpu_utilization_matrix_mult(void (*func)(int, struct Matrix*, struct Matrix*, struct Matrix*, int), int threads, struct Matrix *A, struct Matrix *B, struct Matrix *C, int chunk_size) {
  clock_t start, end;
  double cpu_times_used[RUNS];
  double cpu_time_mean = 0.0;

  for (int i = 0; i < RUNS; i++) {
    start = clock();
    func(threads, A, B , C, chunk_size);
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
  FILE *fp_pi_ap_sch;
  FILE *fp_m_m_sch;
  fp_pi_ap = fopen("pi_approximation.csv", "w");
  fp_m_m = fopen("matrix_multiplication.csv", "w");
  fp_pi_ap_sch = fopen("pi_approximation_schedule.csv", "w");
  fp_m_m_sch = fopen("matrix_multiplication_schedule.csv", "w");

  fprintf(fp_pi_ap, "threads,num steps,sequential,parallel for,parallel for reduction");
  fprintf(fp_m_m, "threads,matrix dimension,sequential,parallel for,parallel for reduction");
  fprintf(fp_pi_ap_sch, "threads,num steps,schedule type,chunk size,static schedule,dynamic schedule");
  fprintf(fp_m_m_sch, "threads,matrix dimension,schedule type,chunk size,static schedule,dynamic schedule");

  long num_steps_array[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
  int pi_approximation_test_runs = sizeof(num_steps_array) / sizeof(num_steps_array[0]);
  // Square matrices for simplicity
  int matrix_dimensions[] = {10, 50, 100, 250};
  int matrix_multiplication_test_runs = sizeof(matrix_dimensions) / sizeof(matrix_dimensions[0]);
  int chunk_sizes[] = {10, 100, 300, 500, 700, 1000};
  const int DEFAULT_CHUNK_SIZE = 1;

  for (int threads = 2; threads < 9; threads = threads + 2) {
    // Pi approximation
    printf("\nStatistics for pi approximation:\n");

    for (int i = 0; i < pi_approximation_test_runs; i++) {
      long num_steps = num_steps_array[i];

      double stats_without_omp = calculate_cpu_utilization_pi_approx(aproximate_pi_without_omp, threads, num_steps, DEFAULT_CHUNK_SIZE);
      double stats_parallel_for = calculate_cpu_utilization_pi_approx(aproximate_pi_parallel_for, threads, num_steps, DEFAULT_CHUNK_SIZE);
      double stats_parallel_for_and_reduction = calculate_cpu_utilization_pi_approx(aproximate_pi_parallel_for_reduction, threads, num_steps, DEFAULT_CHUNK_SIZE);

      printf("Threads: %d Number of steps:%ld\n", threads, num_steps);
      fprintf(fp_pi_ap, "\n%d,%ld,%.6f,%.6f,%.6f", threads, num_steps, stats_without_omp, stats_parallel_for, stats_parallel_for_and_reduction);

      for (int j = 0; j < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); j++) {
        int chunk_size = chunk_sizes[j];
        double stats_parallel_for_static_sch = calculate_cpu_utilization_pi_approx(aproximate_pi_static_schedule, threads, num_steps, chunk_size);
        double stats_parallel_for_dynamic_sch = calculate_cpu_utilization_pi_approx(aproximate_pi_dynamic_schedule, threads, num_steps, chunk_size);

        printf("Threads: %d Number of steps:%ld Chunk size:%d\n", threads, num_steps, chunk_size);
        fprintf(fp_pi_ap_sch, "\n%d,%ld,static,%d,%.6f,%.6f", threads, num_steps, chunk_size, stats_parallel_for_static_sch, stats_parallel_for_dynamic_sch);
      }
    }

    // Matrix multiplication
    printf("\nStatistics for matrix multiplication:\n");

    for(int i = 0; i < matrix_multiplication_test_runs; i++) {
      int matrix_dimension = matrix_dimensions[i];

      struct Matrix* A = create_random_matrix(matrix_dimension, matrix_dimension);
      struct Matrix* B = create_random_matrix(matrix_dimension, matrix_dimension);
      struct Matrix* C = create_random_matrix(matrix_dimension, matrix_dimension);

      double stats_matrix_multiplication_without_omp = calculate_cpu_utilization_matrix_mult(matrix_multiplication_without_omp, threads, A, B, C, DEFAULT_CHUNK_SIZE);
      double stats_matrix_multiplication_parallel_for = calculate_cpu_utilization_matrix_mult(matrix_multiplication_parallel_for, threads, A, B, C, DEFAULT_CHUNK_SIZE);
      double stats_matrix_multiplication_parallel_for_reduce = calculate_cpu_utilization_matrix_mult(matrix_multiplication_parallel_for_reduce, threads, A, B, C, DEFAULT_CHUNK_SIZE);

      printf("Threads: %d Matrix dimension:%d\n", threads, matrix_dimension);
      fprintf(fp_m_m, "\n%d,%d,%.6f,%.6f,%.6f", threads, matrix_dimension, stats_matrix_multiplication_without_omp, stats_matrix_multiplication_parallel_for, stats_matrix_multiplication_parallel_for_reduce);

      for (int j = 0; j < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); j++) {
        int chunk_size = chunk_sizes[j];
        double stats_parallel_for_static_sch = calculate_cpu_utilization_matrix_mult(matrix_multiplication_static_schedule, threads, A, B, C, chunk_size);
        double stats_parallel_for_dynamic_sch = calculate_cpu_utilization_matrix_mult(matrix_multiplication_dynamic_schedule, threads, A, B, C, chunk_size);

        printf("Threads: %d Matrix dimension:%d Chunk size:%d\n", threads, matrix_dimension, chunk_size);
        fprintf(fp_pi_ap_sch, "\n%d,%ld,static,%d,%.6f,%.6f", threads, matrix_dimension, chunk_size, stats_parallel_for_static_sch, stats_parallel_for_dynamic_sch);
      }
    }
  }


}
