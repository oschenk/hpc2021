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

int min(int u, int v) { return u < v ? u : v; }

int at(int n, int i, int j) { return i + j * n; }

short debug = 0;

// C[uw] += A[uv] * B[vw]
// Assumption: the array boundaries are sane.
void compute_for_subblock(int n, double *A, double *B, double *C, int rA,
                          int cA, int rB, int cB, int rC, int cC, int u, int v,
                          int w) {
  if (debug)
    printf("%d | A[%d,%d] B[%d,%d] C[%d,%d] | uv[%dx%d].vw[%dx%d]\n", n, rA, cA,
           rB, cB, rC, cC, u, v, v, w);
  for (int i = 0; i < u; ++i) {
    for (int j = 0; j < w; ++j) {
      double c_ij = C[at(n, rC + i, cC + j)];
      for (int k = 0; k < v; ++k) {
        c_ij += A[at(n, rA + i, cA + k)] * B[at(n, rB + k, cB + j)];
      }
      C[at(n, rC + i, cC + j)] = c_ij;
    }
  }
}

void blocked_dgemm(int n, double *A, double *B, double *C, int block_size) {
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < n; j += block_size) {
      for (int k = 0; k < n; k += block_size) {
        compute_for_subblock(n, A, B, C, i, k, k, j, i, j,
                             min(block_size, n - i), min(block_size, n - k),
                             min(block_size, n - j));
      }
    }
  }
}

void naive_dgemm(int n, double *A, double *B, double *C) {
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) {
      /* Compute C(i,j) */
      double cij = C[at(n, i, j)];
      for (int k = 0; k < n; k++)
        cij += A[at(n, i, k)] * B[at(n, k, j)];
      C[at(n, i, j)] = cij;
    }
}

const char *dgemm_desc = "Optimized dgemm [pratyai/blocked].";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  blocked_dgemm(n, A, B, C, 16);
  // naive_dgemm(n, A, B, C);
  if (debug)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        printf("%+2.1f ", C[at(n, i, j)]);
      }
      printf("\n");
    }
  debug = 0;
}
