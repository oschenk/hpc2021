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

#include <stdio.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define swap(x, y)                                                             \
  {                                                                            \
    double tmp = (x);                                                          \
    (x) = (y);                                                                 \
    (y) = tmp;                                                                 \
  }
static inline int colmajor(int n, int i, int j) { return i + j * n; }
static inline int rowmajor(int n, int i, int j) { return i * n + j; }
static inline void print(int n, double *X, short colmaj) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%+2.2lf ", X[colmaj ? colmajor(n, i, j) : rowmajor(n, i, j)]);
    }
    printf("\n");
  }
  printf("\n");
}
static inline void transpose_square(int n, double *X) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      swap(X[colmajor(n, i, j)], X[rowmajor(n, i, j)]);
    }
  }
}

static void blocked_dgemm(int n, double *A, double *B, double *C,
                          int block_size) {
  // Iterate through all possible valid submatrix combinations.
  // Then we need to compute `C'=C'+A'*B'` for each selection. The naive
  // formula would be:
  //   `C[i][j] += A[i][k] * B[k][j]`
  // However, the matrices are in column-major format (i.e. `X[i+1][j]` is next
  // to `X[i][j]` in memory, but `X[i][j+1]` is far away). So, we want to
  // iterate over `i` in the innermost loop to keep temporaly close accesses
  // also spacially close.
  transpose_square(n, A);
  for (int i = 0; i < n; i += block_size) {
    const int il = min(i + block_size, n);
    for (int j = 0; j < n; j += block_size) {
      const int jl = min(j + block_size, n);
      for (int k = 0; k < n; k += block_size) {
        const int kl = min(k + block_size, n);
        for (int ii = i; ii < il; ++ii) {
          for (int jj = j; jj < jl; ++jj) {
            double c_ij = C[ii + jj * n];
            for (int kk = k; kk < kl; ++kk) {
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

static void naive_dgemm(int n, double *A, double *B, double *C) {
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) {
      /* Compute C(i,j) */
      double cij = C[i + j * n];
      for (int k = 0; k < n; k++)
        cij += A[i + k * n] * B[k + j * n];
      C[i + j * n] = cij;
    }
}

const char *dgemm_desc = "Blocked dgemm [pratyai].";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  // naive_dgemm(n, A, B, C);
  blocked_dgemm(n, A, B, C, 10);
}
