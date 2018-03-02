#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

static constexpr int NUMBER_BENCHMARK_RUNS = 4;

struct BenchResult {
	BenchResult(int n = 0) { measurements.reserve(n); }

	double average() {
		if(measurements.empty()) { return -1; }
		auto sum = 0.0;
		for(auto val : measurements) {
			sum += val;
		}
		return sum / measurements.size();
	}

	double median() {
		if(measurements.empty()) { return -1; }
		std::sort(measurements.begin(), measurements.end());
		if((measurements.size() & 0x1) == 0x1) { // check if we have a odd number of elements
			return measurements[measurements.size() / 2];
		} else { // we have a even number of elements, must average the two middle ones
			return (measurements[measurements.size() / 2 - 1] + measurements[measurements.size() / 2 - 1]) / 2;
		}
	}

	double standard_deviation() {
		if(measurements.size() <= 1) { return 0; }
		auto avg = average();
		auto squared_sum = 0.0;
		for(auto val : measurements) {
			squared_sum += (val - avg) * (val - avg);
		}
		return sqrt(squared_sum / (measurements.size() - 1));
	}

	std::string summary() {
		std::ostringstream os;
		os.precision(1);
		os << std::fixed;
		os << average() << "ms \u00B1 " << standard_deviation() << " (median " << median() << "ms)";
		return os.str();
	}

	std::string raw() {
		std::ostringstream os;
		for(auto val : measurements) {
			std::cout << val << " ";
		}
		return os.str();
	}

	void addMeasurement(double v) {
		if(firstMeasurementOmitted) { measurements.push_back(v); }
		firstMeasurementOmitted = true;
	}

  private:
	std::vector<double> measurements; // milliseconds used for each run of the benchmark
	bool firstMeasurementOmitted = false;
};
