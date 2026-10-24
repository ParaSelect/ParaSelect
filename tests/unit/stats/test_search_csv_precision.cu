#include "benchmark/search_benchmark.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("paraselect-csv-precision-" + std::to_string(getpid()) + ".csv");
    try {
        const double values[] = {1402020720.0, 0.999987654321, 12345.6789012345};
        if (!paraselect::SearchBenchmark::appendCsvRow(path.string(),
                {"n_vectors", "recall", "qps"},
                {{"n_vectors",values[0]}, {"recall",values[1]}, {"qps",values[2]}}))
            throw std::runtime_error("CSV write failed");
        std::ifstream input(path);
        std::string line,field;
        std::getline(input,line);
        if (line != "n_vectors,recall,qps") throw std::runtime_error("CSV header changed");
        std::getline(input,line);
        std::istringstream row(line);
        for (double expected : values) {
            if (!std::getline(row,field,',') || std::stod(field) != expected)
                throw std::runtime_error("CSV lost numeric precision: " + line);
        }
        std::filesystem::remove(path);
        std::cout << "PASS: SPACEV row count and measured doubles round-trip exactly\n";
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
