#!/bin/bash

MATRIX_SIZES="400 600 800 1000 1200 1400 1600 1800 2000 2200 2400"
NUM_WORKERS_VALUES="1 4"
RERUN_BENCHMARK_COUNT=8

set -e

errcho(){ >&2 echo $@; }

for matrix_size in ${MATRIX_SIZES}; do
    errcho "`date` Benchmarking MatSize: ${matrix_size}"

    for i in $(seq 0 ${RERUN_BENCHMARK_COUNT}); do
        for num_workers in ${NUM_WORKERS_VALUES}; do
            HEADER="${matrix_size} ${num_workers}"
            for val in `NUM_WORKERS=${num_workers} ./mm_allscale ${matrix_size} raw`; do
                echo "${HEADER} ALLSCALE ${val}"
            done

            if [ ${num_workers} -eq 0 ]; then
                for val in `./mm_eigen ${matrix_size} raw`; do
                    echo "${HEADER} EIGEN ${val}"
                done
            else
                for val in `OMP_NUM_THREADS=${num_workers} ./mm_eigen ${matrix_size} raw`; do
                    echo "${HEADER} EIGEN ${val}"
                done
            fi
        done
    done
done
