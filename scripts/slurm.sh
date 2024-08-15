#!/bin/bash -l

#SBATCH -J bla_benchmark
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --exclusive

python3 ./build/BLA-main/scripts/benchmark.py $@
