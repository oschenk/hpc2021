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

const char *dgemm_desc = "Naive, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  // TODO: Implement the blocking optimization

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

int at(int n, int i, int j) { return i + j * n; }

// C[uxw] += A[uxv] * B[vxw]
// Assumption: the array boundaries are sane.
void compute_for_subblock(int n, double *A, double *B, double *C, int rA,
                          int cA, int rB, int cB, int rC, int cC, int u, int v,
                          int w) {
  for (int i = 0; i < u; ++i) {
    for (int j = 0; j < w; ++j) {
      double c_ij = C[at(n, rC + i, cC + j)];
      for (int k = 0; k < v; ++v) {
        c_ij += A[at(n, rA + i, cA + k)] * B[at(n, rB + k, cB + j)];
      }
      C[at(n, i, j)] = c_ij;
    }
  }
}

void compute_for_block_combinations(int n, double *A, double *B, double *C,
                                    int block_size) {
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < n; j += block_size) {
      for (int k = 0; k < n; k += block_size) {
        compute_for_subblock(n, A, B, C, i * block_size, k * block_size,
                             k * block_size, j * block_size, i * block_size,
                             j * block_size,
                             min(block_size, n - i * block_size),
                             min(block_size, n - k * block_size),
                             min(block_size, n - j * block_size));
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
      double cij = C[i + j * n];
      for (int k = 0; k < n; k++)
        cij += A[i + k * n] * B[k + j * n];
      C[i + j * n] = cij;
    }
}
