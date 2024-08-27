#!/usr/bin/env python3

import json
import argparse
import copy
from enum import Enum

colors = ["black", "blue", "brown", "cyan", "darkgray", "gray", "green", "lightgray", "lime", "magenta", "olive", "orange", "pink", "purple", "red", "teal", "violet", "yellow"]

def graphToTex(graph):
    global colors
    print("\\begin{tikzpicture}[scale=0.6]")
    print("\\begin{{axis}}[xlabel={}, ylabel={}, xmode={}, xmin={}, ymin=0, legend pos=outer north east]".format(graph.xlabel, graph.ylabel, graph.xmode, graph.xmin))
    color = 0
    legend = []
    keys = list(graph.y.keys())
    keys.sort()
    for key in keys:
        compiler = key
        legend.append(compiler)
        print("\\addplot[color={},mark=x] coordinates {{".format(colors[color]))
        color += 1
        for value in graph.y[key]:
            print("({}, {})".format(value[0], value[1]))
        print("};")
    print("\\legend{{{}}}".format(",".join(legend)))
    print("\\end{axis}")
    print("\\end{tikzpicture}")

class GraphType(Enum):
    THREADS = 1
    PERFORMANCE = 2

class Graph:
    def __init__(self, graphType, name):
        self.ylabel = "Runtime (ns)"
        self.name = name
        self.graphType = graphType
        self.xmin = 1
        if graphType == GraphType.THREADS:
            self.xlabel = "Threads"
        else:
            self.xlabel = "Matrix Size"
        self.y = {}
        self.xmode = "normal"

    def insert(self, libName, x, y):
        if not libName in self.y:
            self.y[libName] = []
        values = self.y[libName]
        # assert(x == 0 or x == self.x[len(values)])
        values.append((int(x), int(y)))

    def apply(self, func):
        for key in self.y:
            values = self.y[key]
            for i in range(len(values)):
                values[i] = func(values[i])

    def __str__(self):
        return self.name

class Result:
    def __init__(self, data):
        self.graphs = {}
        self.values = {}
        for benchmarkName in data:
            benchmarkData = data[benchmarkName]
            for resultData in benchmarkData:
                libName = resultData["name"]
                for result in resultData["results"]:
                    numThreads = result["num_threads"] if "num_threads" in result else None
                    resultGBenchmark = result["benchmark"]
                    for gBenchmark in resultGBenchmark["benchmarks"]:
                        benchmarkSize = gBenchmark["name"].split("/")[1]
                        graphType = GraphType.THREADS if numThreads is not None else GraphType.PERFORMANCE
                        if graphType == GraphType.THREADS:
                            xvalue = numThreads
                            name = "{}/{}".format(benchmarkName, benchmarkSize)
                        else:
                            xvalue = benchmarkSize
                            name = benchmarkName
                        graph = self.getGraph(graphType, name)
                        graph.insert(libName, xvalue, gBenchmark["real_time"])
                        assert(gBenchmark["time_unit"] == "ns")

    def getGraph(self, graphType, name):
        if (graphType, name) in self.graphs:
            return self.graphs[(graphType, name)]
        self.graphs[(graphType, name)] = Graph(graphType, name)
        return self.graphs[(graphType, name)]

    def summary(self, forceRuntime):
        print("\\documentclass{article}")
        print("\\usepackage{tikz,pgfplots}")
        print("\\begin{document}")
        print("\\section{Benchmarks}")
        for graphKey in self.graphs:
            graph = self.graphs[graphKey]

            if graph.graphType == GraphType.THREADS:
                fastest = None
                for key in graph.y:
                    v = graph.y[key][0][1]
                    if not fastest or v < fastest:
                        fastest = v

                # speed up
                speedup_graph = copy.deepcopy(graph)
                speedup_graph.apply(lambda value: (value[0], fastest / value[1]))
                speedup_graph.ylabel = "Speed up"

                # efficiency
                efficiency_graph = copy.deepcopy(graph)
                efficiency_graph.apply(lambda value: (value[0], (fastest / value[1]) / value[0]))
                efficiency_graph.ylabel = "Efficiency"

                print("\\section{{{}}}".format(graph.name))
                graphToTex(graph)
                graphToTex(speedup_graph)
                graphToTex(efficiency_graph)
            else:
                if not forceRuntime:
                    if "add" in graph.name:
                        graph.apply(lambda value: (value[0], (3 * value[0] * value[0] * value[0]) / value[1]))
                        graph.ylabel = "GFLOPS"
                    elif "transpose" in graph.name:
                        graph.apply(lambda value: (value[0], 1000 * (value[0] * value[0] * 8) / value[1]))
                        graph.ylabel = "MB/s"
                    elif "mm" in graph.name:
                        graph.apply(lambda value: (value[0], (2 * value[0] * value[0] * value[0] - value[0] * value[0]) / value[1]))
                        graph.ylabel = "GFLOPS"
                graph.xmode = "log"
                graphToTex(graph)
        print("\\end{document}")

def calculate(json_data, forceRuntime):
    result = Result(json_data)
    result.summary(forceRuntime)

def parseArgs():
    parser = argparse.ArgumentParser(description="Plot test results")
    parser.add_argument("--in", dest="in_file", action="store", help="File to read the result from", default="result.json")
    parser.add_argument("--runtime", dest="runtime", action="store_true", help="Force graph to show runtime")

    return parser.parse_args()

def main():
    args = parseArgs()
    with open(args.in_file, 'r') as in_file:
        json_data = json.load(in_file)
        calculate(json_data, args.runtime)

if __name__ == "__main__":
    main()
