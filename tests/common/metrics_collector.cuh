#ifndef METRIC_COLLECTOR_H
#define METRIC_COLLECTOR_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cstring>
#include "test_config.h"
// ============================================================================
// Metrics  (Metrics Collector)
// ============================================================================

/**
 * Metrics
 *
 *
 * :
 * MetricsCollector metrics;
 * metrics.set_columns("batch", "k", "gpu_ms", "cpu_ms", "speedup");
 *
 * PARAM_2D(batch, (100, 1000), k, (16, 64)) {
 *     test_function(batch, k, &metrics);  //  metrics.add_row()
 * }
 *
 * metrics.print_table();
 */
class MetricsCollector {
private:
    std::vector<std::string> column_names_;
    std::vector<std::vector<double>> rows_;
    std::vector<int> column_widths_;
    bool columns_set_ = false;
    int num_repeats_ = 10;        /*  10  */
    int base_seed_ = 42;          /*  */

    std::string csv_path_ = std::string(PROJECT_PATH) + std::string(CSV_PATH);

    /*  */
    void calculate_column_widths() {
        column_widths_.resize(column_names_.size());

        /*  */
        for (size_t i = 0; i < column_names_.size(); i++) {
            column_widths_[i] = column_names_[i].length();
        }

        /*  */
        for (const auto& row : rows_) {
            for (size_t i = 0; i < row.size() && i < column_widths_.size(); i++) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3) << row[i];
                int width = oss.str().length();
                if (width > column_widths_[i]) {
                    column_widths_[i] = width;
                }
            }
        }

        /*  6 */
        for (auto& w : column_widths_) {
            if (w < 6) w = 6;
        }
    }

    /*  */
    void print_separator() const {
        std::cout << "+";
        for (size_t i = 0; i < column_widths_.size(); i++) {
            std::cout << std::string(column_widths_[i] + 2, '-') << "+";
        }
        std::cout << "\n";
    }

public:
    MetricsCollector() = default;

    /**
     * ()
     */
    template<typename... Args>
    void set_columns(Args&&... names) {
        column_names_ = {std::forward<Args>(names)...};
        columns_set_ = true;
    }

    /**
     * ()-
     */
    template<typename... Args>
    void add_row(Args&&... values) {
        if (!columns_set_) {
            std::cerr << "Error: Must call set_columns() before add_row()\n";
            return;
        }

        std::vector<double> row;
        row.reserve(sizeof...(values));

        /*  double  row */
        (row.push_back(static_cast<double>(values)), ...);

        if (row.size() != column_names_.size()) {
            std::cerr << "Error: Row has " << row.size()
                      << " values but " << column_names_.size()
                      << " columns expected\n";
            return;
        }

        rows_.push_back(std::move(row));
    }

    /**
     * ()
     *
     * :
     * metrics.add_row_averaged([&]() -> std::vector<double> {
     *     // ,
     *     double gpu_ms = benchmark(...);
     *     return {batch, k, gpu_ms, speedup};
     * });
     *
     * @param test_func , std::vector<double>
     * @return
     */
    template<typename Func>
    std::vector<double> add_row_averaged(Func test_func) {
        if (!columns_set_) {
            std::cerr << "Error: Must call set_columns() before add_row_averaged()\n";
            return {};
        }

        std::vector<double> sum_row(column_names_.size(), 0.0);
        int valid_runs = 0;

        for (int i = 0; i < num_repeats_; i++) {
            srand(base_seed_ + i);

            std::vector<double> row = test_func();

            if (row.size() != column_names_.size()) {
                std::cerr << "Warning: Row size mismatch in add_row_averaged (run " << i << ")\n";
                continue;
            }

            for (size_t j = 0; j < row.size(); j++) {
                sum_row[j] += row[j];
            }
            valid_runs++;
        }

        if (valid_runs == 0) {
            std::cerr << "Error: No valid runs in add_row_averaged\n";
            return sum_row;
        }

        for (auto& val : sum_row) {
            val /= valid_runs;
        }

        rows_.push_back(sum_row);

        return sum_row;
    }

    /**
     *
     */
    void set_num_repeats(int n) {
        if (n > 0) {
            num_repeats_ = n;
        } else {
            std::cerr << "Warning: num_repeats must be positive, using default (10)\n";
        }
    }

    /**
     *
     */
    void set_base_seed(int seed) {
        base_seed_ = seed;
    }

    /**
     *
     */
    int get_num_repeats() const {
        return num_repeats_;
    }

    /**
     * ( metrics )
     */
    std::vector<double> get_last_row() const {
        if (rows_.empty()) {
            return {};
        }
        return rows_.back();
    }

    /**
     *
     */
    std::vector<double> get_row(size_t idx) const {
        if (idx >= rows_.size()) {
            return {};
        }
        return rows_[idx];
    }

    /**
     *
     */
    void print_table() {
        if (!columns_set_ || column_names_.empty()) {
            std::cout << "No metrics to display.\n";
            return;
        }

        COUT_ENDL("  Metrics: ");

        calculate_column_widths();

        /*  */
        print_separator();
        std::cout << "|";
        for (size_t i = 0; i < column_names_.size(); i++) {
            std::cout << " " << std::left << std::setw(column_widths_[i])
                      << column_names_[i] << " |";
        }
        std::cout << "\n";
        print_separator();

        /*  */
        for (const auto& row : rows_) {
            std::cout << "|";
            for (size_t i = 0; i < row.size(); i++) {
                std::cout << " " << std::right << std::setw(column_widths_[i])
                          << std::fixed << std::setprecision(3) << row[i] << " |";
            }
            std::cout << "\n";
        }
        print_separator();

        /*  */
        std::cout << "Total rows: " << rows_.size();
        if (num_repeats_ > 1) {
            std::cout << " (averaged over " << num_repeats_ << " runs)";
        }
        std::cout << "\n";
    }

    /**
     * ()
     */
    void clear_rows() {
        rows_.clear();
    }

    /**
     * ()
     */
    void reset() {
        column_names_.clear();
        rows_.clear();
        column_widths_.clear();
        columns_set_ = false;
    }

    /**
     *
     */
    size_t size() const {
        return rows_.size();
    }

    /**
     *
     */
    double mean(size_t col_idx) const {
        if (col_idx >= column_names_.size() || rows_.empty()) {
            return 0.0;
        }

        double sum = 0.0;
        for (const auto& row : rows_) {
            if (col_idx < row.size()) {
                sum += row[col_idx];
            }
        }
        return sum / rows_.size();
    }

    /**
     * ()
     */
    double mean(const std::string& col_name) const {
        for (size_t i = 0; i < column_names_.size(); i++) {
            if (column_names_[i] == col_name) {
                return mean(i);
            }
        }
        std::cerr << "Error: Column '" << col_name << "' not found\n";
        return 0.0;
    }

    /**
     *  CSV
     */
    void export_csv(const std::string& filename) const {
        // (C++14 )
        struct stat info;
        if (stat(PROJECT_PATH, &info) != 0) {
            std::cerr << "Error: PROJECT_PATH does not exist: " << PROJECT_PATH << "\n";
            std::cerr << "Cannot export CSV file. Please ensure the project directory exists.\n";
            return;
        }

        if (!(info.st_mode & S_IFDIR)) {
            std::cerr << "Error: PROJECT_PATH is not a directory: " << PROJECT_PATH << "\n";
            return;
        }

        //  CSV (:,)
        // ,,
        if (stat(csv_path_.c_str(), &info) != 0) {
            std::cerr << "Warning: CSV output directory does not exist: " << csv_path_ << "\n";
            std::cerr << "Attempting to create file anyway...\n";
        }

        std::string file_path = csv_path_ + filename;
        std::ofstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << file_path << "\n";
            return;
        }

        /*  */
        for (size_t i = 0; i < column_names_.size(); i++) {
            file << column_names_[i];
            if (i < column_names_.size() - 1) file << ",";
        }
        file << "\n";

        /*  */
        for (const auto& row : rows_) {
            for (size_t i = 0; i < row.size(); i++) {
                file << std::fixed << std::setprecision(6) << row[i];
                if (i < row.size() - 1) file << ",";
            }
            file << "\n";
        }

        file.close();
        std::cout << "Metrics exported to " << file_path << "\n";
    }
};

#endif
