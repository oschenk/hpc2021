/*
    Please include compiler name below (you may also include any other modules
you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You
can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a
$(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a
-Wl,--end-group -lpthread -lm

*/

/*
 * How to perf:
 * 1. `make`
 * 2. `perf record -e
 * cycles,LLC-loads,LLC-load-misses,cache-references,cache-misses,branch-misses
 * ./benchmark-blocked`
 * 3. `perf report`
 */

#ifndef BLOCKSIZE
#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64 // Typically.
#endif
#define BLOCKSIZE (CACHE_LINE_SIZE / sizeof(double))
#endif

#include <stdio.h>
#include <stdlib.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define swap(x, y)                                                             \
  {                                                                            \
    register double tmp = (x);                                                 \
    (x) = (y);                                                                 \
    (y) = tmp;                                                                 \
  }
static inline void print(int n, double *X, short colmaj) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%+2.2lf ", X[i + j * n]);
    }
    printf("\n");
  }
  printf("\n");
}
static inline void transpose_square(int n, double *X) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      swap(X[i + j * n], X[i * n + j]);
    }
  }
}
static inline void print_dist(double **last, double *now,
                              const char *const msg) {
  printf("%s: %ld\n", msg, now - *last);
  *last = now;
}

// #define DEBUG 1

static void blocked_dgemm(int n, double *A, double *B, double *C) {
#ifdef DEBUG
  static double *last_A = NULL;
  static double *last_B = NULL;
  static double *last_C = NULL;
#endif
  transpose_square(n, A);
  // Partition each matrices into smaller subblocks.
  for (int i = 0; i < n; i += BLOCKSIZE) {
    const int ilim = min(n, i + BLOCKSIZE);
    for (int j = 0; j < n; j += BLOCKSIZE) {
      const int jlim = min(n, j + BLOCKSIZE);
      for (int k = 0; k < n; k += BLOCKSIZE) {
        const int klim = min(n, k + BLOCKSIZE);
        // And compute C_ij += A_ik * B_kj;
        for (int ii = i; ii < ilim; ++ii) {
          for (int jj = j; jj < jlim; ++jj) {
#ifdef DEBUG
            print_dist(&last_C, &C[ii + jj * n], "C");
#endif
            double c_ij = C[ii + jj * n];
            for (int kk = k; kk < klim; ++kk) {
#ifdef DEBUG
              print_dist(&last_A, &A[ii * n + kk], "A");
              print_dist(&last_B, &B[kk + jj * n], "B");
#endif
              c_ij += A[ii * n + kk] * B[kk + jj * n];
            }
            C[ii + jj * n] = c_ij;
          }
        }
      }
    }
  }
  transpose_square(n, A);
}

const char *dgemm_desc = "Blocked dgemm [pratyai].";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  blocked_dgemm(n, A, B, C);
  // if (n >= 480) exit(0);
#ifdef DEBUG
  exit(0);
#endif
}
