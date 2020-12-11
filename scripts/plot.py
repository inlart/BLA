#!/usr/bin/env python3

import json
import argparse

class Graph:
    def __init__(self, name, size, x):
        self.name = name
        self.size = size
        self.x = x
        self.y = {}

    def insert(self, libName, x, y):
        if not libName in self.y:
            self.y[libName] = []
        values = self.y[libName]
        assert(x == self.x[len(values)])
        values.append(y)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def __str__(self):
        return self.name + "/" + self.size

class Result:
    def __init__(self, data):
        self.graphs = {}
        # Calculate max threads
        self.max = 1
        for benchmarkName in data:
            for resultData in data[benchmarkName]:
                for result in resultData["results"]:
                    if result["num_threads"] > self.max:
                        self.max = result["num_threads"]
        self.values = {}
        for benchmarkName in data:
            benchmarkData = data[benchmarkName]
            for resultData in benchmarkData:
                libName = resultData["name"]
                for result in resultData["results"]:
                    numThreads = result["num_threads"]
                    resultGBenchmark = result["results"]
                    for gBenchmark in resultGBenchmark["benchmarks"]:
                        benchmarkSize = gBenchmark["name"].split("/")[1]
                        graph = self.getGraph(benchmarkName, benchmarkSize)
                        graph.insert(libName, numThreads, gBenchmark["real_time"])
                        assert(gBenchmark["time_unit"] == "ns")

    def getGraph(self, name, size):
        if (name, size) in self.graphs:
            return self.graphs[(name, size)]
        self.graphs[(name, size)] = Graph(name, size, list(range(1, self.max + 1)))
        return self.graphs[(name, size)]

    def summary(self):
        for graphKey in self.graphs:
            graph = self.graphs[graphKey]
            print(graph)
            print(graph.getX())
            yValues = graph.getY()
            for yValue in yValues:
                print(yValue + " "  + str(yValues[yValue]))

def calculate(json_data):
    result = Result(json_data)
    result.summary()

def parseArgs():
    parser = argparse.ArgumentParser(description="Plot test results")
    parser.add_argument("--in", dest="in_file", action="store", help="File to read the result from", default="result.json")

    return parser.parse_args()

def main():
    args = parseArgs()
    with open(args.in_file, 'r') as in_file:
        json_data = json.load(in_file)
        calculate(json_data)

if __name__ == "__main__":
    main()
