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
  // Iterate through all possible valid submatrix combinations.
  // Then we need to compute `C'=C'+A'*B'` for each selection. The naive
  // formula would be:
  //   `C[i][j] += A[i][k] * B[k][j]`
  // However, the matrices are in column-major format (i.e. `X[i+1][j]` is next
  // to `X[i][j]` in memory, but `X[i][j+1]` is far away). So, we want to
  // iterate over `i` in the innermost loop to keep temporaly close accesses
  // also spacially close.
  for (register int k = 0; k < n; k += block_size) {
    const register int kl = min(k + block_size, n);
    for (register int j = 0; j < n; j += block_size) {
      const register int jl = min(j + block_size, n);
      for (register int i = 0; i < n; i += block_size) {
        const register int il = min(i + block_size, n);
        for (register int kk = k; kk < kl; ++kk) {
          register double *const a_start = A + kk * n;
          for (register int jj = j; jj < jl; ++jj) {
            register double *const c_start = C + jj * n;
            const register double b_kj = B[kk + jj * n];
            for (register int ii = i; ii < il; ++ii) {
              c_start[ii] += a_start[ii] * b_kj;
            }
          }
        }
      }
    }
  }
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
  // single_file_dgemm(n, A, B, C);
  blocked_dgemm(n, A, B, C, 10);
}
