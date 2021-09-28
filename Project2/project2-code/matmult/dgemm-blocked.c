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
inline void transpose_square(const int n, double *X) {
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

void force(double *ptr) { asm volatile("" : "=m"(*ptr) : "r"(*ptr)); }

void blocked_dgemm(const int n, double *A, double *B, double *C) {
  const int blocksize = BLOCKSIZE;
  transpose_square(n, A);
  // Partition each matrices into smaller subblocks.
  for (int i = 0; i < n; i += blocksize) {
    const int ilim = min(n, i + blocksize);
    for (int j = 0; j < n; j += blocksize) {
      const int jlim = min(n, j + blocksize);
      // Load B, C subblocks into cache by quickly scanning through them.
      // Loading A doesn't have significant impact.
      for (int jj = j; jj < jlim; ++jj) {
        force(&B[jj * n]);
        force(&C[jj * n]);
      }
      for (int k = 0; k < n; k += blocksize) {
        const int klim = min(n, k + blocksize);

        // And partially compute for the current subblocks: C_ij += A_ik * B_kj.
        for (int ii = i; ii < ilim; ++ii) {
          for (int jj = j; jj < jlim; ++jj) {
            double c_ij = C[ii + jj * n];
            for (int kk = k; kk < klim; ++kk) {
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
#ifdef DEBUG
  exit(0);
#endif
}
