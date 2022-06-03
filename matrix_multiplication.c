#ifndef MATRIX_MULTIPLICATION
#define MATRIX_MULTIPLICATION

#define MATRIX_MULTIPLICATION_NUM_ARGS 3

#include <stdlib.h>

struct Matrix {
  int rows;
  int columns;
  double *data;
};

// Helper function
struct Matrix* create_random_matrix(int rows, int columns) {
  struct Matrix *matrix = malloc(sizeof(struct Matrix));
  matrix->rows = rows;
  matrix->columns = columns;
  matrix->data = malloc(sizeof(double) * rows * columns);
  for (int i = 0; i < rows * columns; i++) {
    matrix->data[i] = (double) rand() / RAND_MAX;
  }
  return matrix;
}

void matrix_multiplication_without_omp(int threads, struct Matrix *A, struct Matrix *B, struct Matrix *C) {
  int i, j, k;
  double sum;

  for (i = 0; i < A->rows; i++) {
    for (j = 0; j < B->columns; j++) {
      sum = 0.0;
      for (k = 0; k < A->columns; k++) {
        sum += A->data[i * A->columns + k] * B->data[k * B->columns + j];
      }
      C->data[i * C->columns + j] = sum;
    }
  }
}

void matrix_multiplication_parallel_for(int threads, struct Matrix *A, struct Matrix *B, struct Matrix *C) {
  int i, j, k;
  double sum;

  #pragma omp parallel for private(i)
  for (i = 0; i < A->rows; i++) {
    #pragma omp parallel for private(j)
    for (j = 0; j < B->columns; j++) {
      sum = 0.0;
      #pragma omp parallel for private(k)
      for (k = 0; k < A->columns; k++) {
        #pragma omp atomic
        sum += A->data[i * A->columns + k] * B->data[k * B->columns + j];
      }
      C->data[i * C->columns + j] = sum;
    }
  }
}

void matrix_multiplication_parallel_for_reduce(int threads, struct Matrix *A, struct Matrix *B, struct Matrix *C) {
  int i, j, k;
  double sum;

  #pragma omp parallel for private(i)
  for (i = 0; i < A->rows; i++) {
    #pragma omp parallel for private(j)
    for (j = 0; j < B->columns; j++) {
      sum = 0.0;
      #pragma omp parallel for private(k) reduction(+:sum)
      for (k = 0; k < A->columns; k++) {
        sum += A->data[i * A->columns + k] * B->data[k * B->columns + j];
      }
      C->data[i * C->columns + j] = sum;
    }
  }
}

#endif
