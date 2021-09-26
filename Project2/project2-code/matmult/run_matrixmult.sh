#!/bin/bash -l

#SBATCH --job-name=matrixmult
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=matrixmult-%j.out
#SBATCH --error=matrixmult-%j.err

# load modules
if command -v module 1>/dev/null 2>&1; then
   module load gcc/10.1.0 intel-mkl/2020.1.217-gcc-10.1.0-qsctnr6 gnuplot
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "==== benchmark-naive ======================"
./benchmark-naive | tee timing_basic_dgemm.data
echo
echo "==== benchmark-blas ======================="
./benchmark-blas | tee timing_blas_dgemm.data
echo
echo "==== benchmark-blocked ===================="
./benchmark-blocked | tee timing_blocked_dgemm.data
echo
echo "==== benchmark-blocked/2 ===================="
./benchmark-blocked-2 | tee timing_blocked_dgemm_2.data
echo
echo "==== benchmark-blocked/4 ===================="
./benchmark-blocked-4 | tee timing_blocked_dgemm_4.data
echo
echo "==== benchmark-blocked/8 ===================="
./benchmark-blocked-8 | tee timing_blocked_dgemm_8.data
echo
echo "==== benchmark-blocked/16 ===================="
./benchmark-blocked-16 | tee timing_blocked_dgemm_16.data
echo
echo "==== benchmark-blocked/32 ===================="
./benchmark-blocked-32 | tee timing_blocked_dgemm_32.data
echo
echo "==== benchmark-blocked/64 ===================="
./benchmark-blocked-64 | tee timing_blocked_dgemm_64.data
echo
echo "==== benchmark-blocked/128 ===================="
./benchmark-blocked-128 | tee timing_blocked_dgemm_128.data
echo
echo "==== plot results ========================="
gnuplot timing.gp

ps2pdf timing.ps
