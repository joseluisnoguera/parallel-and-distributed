#ifndef PI_APROXIMATION
#define PI_APROXIMATION

#define PI_APROXIMATION_NUM_ARGS 1

void aproximate_pi_without_omp(int threads, long num_steps, int chunk_size) {
  int i;
  double x, pi, sum = 0.0;

  double step = 1.0 / (double) num_steps;

  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

void aproximate_pi_parallel_for(int threads, long num_steps, int chunk_size) {
  omp_set_num_threads(threads);
  int i;
  double x, pi, sum = 0.0;

  double step = 1.0 / (double) num_steps;

  #pragma omp parallel for private(x, i)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    double result = 4.0 / (1.0 + x * x);
    #pragma omp atomic
    sum += result;
  }

  pi = step * sum;
}

void aproximate_pi_parallel_for_reduction(int threads, long num_steps, int chunk_size) {
  omp_set_num_threads(threads);
  int i;
  double x, pi, sum = 0.0;

  double step = 1.0 / (double) num_steps;

  #pragma omp parallel for private(x, i) reduction(+:sum)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

void aproximate_pi_static_schedule(int threads, long num_steps, int chunk_size) {
  omp_set_num_threads(threads);
  int i;
  double x, pi, sum = 0.0;

  double step = 1.0 / (double) num_steps;

  #pragma omp parallel for schedule(static, chunk_size) private(x, i) reduction(+:sum)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}

void aproximate_pi_dynamic_schedule(int threads, long num_steps, int chunk_size) {
  omp_set_num_threads(threads);
  int i;
  double x, pi, sum = 0.0;

  double step = 1.0 / (double) num_steps;

  #pragma omp parallel for schedule(dynamic, chunk_size) private(x, i) reduction(+:sum)
  for (i=0; i < num_steps; i++) {
    x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  pi = step * sum;
}


#endif
