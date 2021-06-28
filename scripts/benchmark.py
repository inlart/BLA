#!/usr/bin/env python3

import argparse
import os
import psutil
import subprocess
import json

def parseArgs():
    parser = argparse.ArgumentParser(description="Execute benchmarks with differnt number of threads/workers")
    parser.add_argument("--path", dest="benchmark_path", action="store", help="Path that contains the benchmark executables", default=".")
    parser.add_argument("--out", dest="out_file", action="store", help="File to write the result to", default="result.json")
    parser.add_argument("--list", dest="list", action="store_true", help="List available benchmarks")
    return parser.parse_args()

def runBenchmark(filename, path, cpu_count):
    benchmark_split = filename.split("_")
    benchmark_name = benchmark_split[1]
    benchmark_lib = benchmark_split[-1]
    result = {}
    result["name"] = benchmark_lib
    result["results"] = []
    print("Running benchmark {} for library {}.".format(benchmark_name, benchmark_lib))
    for num_threads in range(1, cpu_count):
        os.environ["NUM_WORKERS"] = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        val = subprocess.check_output([path + "/" + filename, '--benchmark_format=json'])
        benchmark = {}
        benchmark["num_threads"] = num_threads
        benchmark["benchmark"] = json.loads(val)
        result["results"].append(benchmark)
    return (benchmark_name, result)

def getBenchmarksPrefix(path, prefix):
    executables = filter(lambda s: s.startswith(prefix) and os.access(path + "/" + s, os.X_OK) and os.path.isfile(path + "/" + s), os.listdir(path))
    return list(executables)

def getBenchmarks(path):
    return getBenchmarksPrefix(path, "benchmark_")

def main():
    args = parseArgs()

    benchmarks = getBenchmarks(args.benchmark_path)

    if args.list:
        print('\n'.join(benchmarks))
        return
    
    cpu_count = psutil.cpu_count()
    result = {}
    for benchmark in benchmarks:
        name, results = runBenchmark(benchmark, args.benchmark_path, cpu_count)
        if name not in result:
            result[name] = []
        result[name].append(results)
    with open(args.out_file, "w") as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    main()
