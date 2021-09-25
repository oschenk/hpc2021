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

static inline int min(int u, int v) { return u < v ? u : v; }
static inline void swap(double *u, double *v) {
  register const double tmp = *u;
  *u = *v;
  *v = tmp;
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
      swap(&X[colmajor(n, i, j)], &X[rowmajor(n, i, j)]);
    }
  }
}

// C[uw] += A[uv] * B[vw]
// A in row major, B, C in column major.
static inline void compute_for_block_combination(int n, double *A, double *B,
                                                 double *C, int rA, int cA,
                                                 int rB, int cB, int rC, int cC,
                                                 int u, int v, int w) {
  for (int i = 0; i < u; ++i) {
    const int a_start = rowmajor(n, rA + i, cA);
    for (int j = 0; j < w; ++j) {
      const int b_start = colmajor(n, rB, cB + j);
      const int c_at = colmajor(n, rC + i, cC + j);
      register double c_ij = C[c_at];
      for (int k = 0; k < v; ++k) {
        c_ij += A[a_start + k] * B[b_start + k];
      }
      C[c_at] = c_ij;
    }
  }
}

static void blocked_dgemm(int n, double *A, double *B, double *C,
                          int block_size) {
  transpose_square(n, A);
  for (int i = 0; i < n; i += block_size) {
    const register int u = min(block_size, n - i);
    for (int j = 0; j < n; j += block_size) {
      const register int w = min(block_size, n - j);
      for (int k = 0; k < n; k += block_size) {
        const register int v = min(block_size, n - k);
        compute_for_block_combination(n, A, B, C, i, k, k, j, i, j, u, v, w);
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
      double cij = C[colmajor(n, i, j)];
      for (int k = 0; k < n; k++)
        cij += A[colmajor(n, i, k)] * B[colmajor(n, k, j)];
      C[colmajor(n, i, j)] = cij;
    }
}

static void single_file_dgemm(int n, double *A, double *B, double *C) {
  transpose_square(n, A);
  for (int i = 0; i < n; ++i) {
    const int a_start = rowmajor(n, i, 0);
    for (int j = 0; j < n; ++j) {
      const int b_start = colmajor(n, 0, j);
      int c_at = colmajor(n, i, j);
      double cij = C[c_at];
      for (int k = 0; k < n; k++) {
        cij += A[a_start + k] * B[b_start + k];
      }
      C[c_at] = cij;
    }
  }
  transpose_square(n, A);
}

const char *dgemm_desc = "Optimized dgemm [pratyai/blocked].";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  // naive_dgemm(n, A, B, C);
  // single_file_dgemm(n, A, B, C);
  blocked_dgemm(n, A, B, C, 10);
}
