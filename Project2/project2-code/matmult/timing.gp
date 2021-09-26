set title "NxN matrix-matrix-multiplication on 4-Core Intel(R) Xeon(R) CPU E3-1585L v5 @ 3.00GHz"
set xlabel "Matrix size (N)"
set ylabel "Performance (GFlop/s)"
set grid
set logscale y 10

set terminal postscript color "Helvetica" 14
set output "timing.ps"

# set terminal png color "Helvetica" 14
# set output "timing.png"

# plot "timing.data" using 2:4 title "square_dgemm" with linespoints


# For performance comparisons

plot "timing_basic_dgemm.data"   using 2:4 title "Naive dgemm" with linespoints, \
     "timing_blocked_dgemm.data" using 2:4 title "Blocked dgemm" with linespoints, \
     "timing_blas_dgemm.data"   using 2:4 title "MKL blas dgemm" with linespoints, \
     "timing_blocked_dgemm_2.data" using 2:4 title "Blocked dgemm/2" with linespoints, \
     "timing_blocked_dgemm_4.data" using 2:4 title "Blocked dgemm/4" with linespoints, \
     "timing_blocked_dgemm_8.data" using 2:4 title "Blocked dgemm/8" with linespoints, \
     "timing_blocked_dgemm_16.data" using 2:4 title "Blocked dgemm/16" with linespoints, \
     "timing_blocked_dgemm_32.data" using 2:4 title "Blocked dgemm/32" with linespoints, \
     "timing_blocked_dgemm_64.data" using 2:4 title "Blocked dgemm/64" with linespoints, \
     "timing_blocked_dgemm_128.data" using 2:4 title "Blocked dgemm/128" with linespoints,
