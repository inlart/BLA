#!/usr/bin/env python3

import argparse
import os
import sys
import psutil
import subprocess
import json
import re

def parseArgs():
    parser = argparse.ArgumentParser(description="Execute benchmarks with different number of threads/workers")
    parser.add_argument("--path", dest="benchmark_path", action="store", help="Path that contains the benchmark executables", default=".")
    parser.add_argument("--out", dest="out_file", action="store", help="File to write the result to", default="result.json")
    parser.add_argument("--list", dest="list", action="store_true", help="List available benchmarks")
    parser.add_argument("--filter", dest="filter", nargs="+", help="Filter benchmarks by name", default=[])
    parser.add_argument("--no-threads", dest="no_threads", action="store_true", help="Disable threads for benchmarks")
    return parser.parse_args()

def runBenchmark(filename, path, threads):
    benchmark_split = filename.split("_")
    benchmark_name = benchmark_split[1]
    benchmark_lib = " ".join(benchmark_split[2:])
    result = {}
    result["name"] = benchmark_lib
    result["results"] = []
    print("Running benchmark {} ({}/{}) for library {}.".format(benchmark_name, path, filename, benchmark_lib))
    for num_threads in threads:
        if num_threads > 0:
            os.environ["NUM_WORKERS"] = str(num_threads)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
        val = subprocess.check_output(["{}/{}".format(path, filename), '--benchmark_format=json'])
        benchmark = {}
        if num_threads > 0:
            benchmark["num_threads"] = num_threads
        benchmark["benchmark"] = json.loads(val)
        result["results"].append(benchmark)
    return (benchmark_name, result)

def getBenchmarksPrefix(path, prefix):
    executables = filter(lambda s: s.startswith(prefix) and os.access(path + "/" + s, os.X_OK) and os.path.isfile(path + "/" + s), os.listdir(path))
    return list(executables)

def getBenchmarks(path):
    return getBenchmarksPrefix(path, "benchmark_")

def runFilter(benchmarks, patterns):
    filtered_benchmarks = []
    progs = []
    for pattern in patterns:
        try:
            prog = re.compile(pattern)
            progs.append(prog)
        except re.error as err:
            print("Invalid filter {}: {}".format(pattern, err))
            sys.exit(1)

    for benchmark in benchmarks:
        for prog in progs:
            if prog.match(benchmark):
                filtered_benchmarks.append(benchmark)
                continue
    return filtered_benchmarks

def main():
    args = parseArgs()

    benchmarks = getBenchmarks(args.benchmark_path)

    if len(args.filter) > 0:
        benchmarks = runFilter(benchmarks, args.filter)

    if args.list:
        print('\n'.join(benchmarks))
        return

    threads = [0]
    if not args.no_threads:
        if "SLURM_CPUS_PER_TASK" in os.environ:
            threads = range(1, int(os.getenv("SLURM_CPUS_PER_TASK")))
        else:
            threads = range(1, psutil.cpu_count())

    result = {}
    for benchmark in benchmarks:
        name, results = runBenchmark(benchmark, args.benchmark_path, threads)
        if name not in result:
            result[name] = []
        result[name].append(results)
    with open(args.out_file, "w") as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    main()
