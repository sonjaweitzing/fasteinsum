// File: tests/test_bmm.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "catch.hpp"
#include "bmm.h"

using namespace std;
using namespace std::chrono;

void generate_random_matrix(vector<double> &matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

double time_function(void (*func)(const double*, const double*, double*, int, int, int, int),
                     const double *a, const double *b, double *c,
                     int batch_dim, int a_rows, int b_cols, int a_cols) {
    auto start = high_resolution_clock::now();
    func(a, b, c, batch_dim, a_rows, b_cols, a_cols);
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    return elapsed.count();
}

bool compare_matrices(const vector<double> &a, const vector<double> &b) {
    const double epsilon = 1e-6;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

// Testing bmm optimized with blocking and kernel
// call: ./tests/catch_tests_bmm -c "[bmm_opt]"
// call: ./tests/catch_tests_bmm -c "Timing and correctness of bmm optimized"

void bmm_blas_wrapper(const double *a, const double *b, double *c,
                      int batch_dim, int a_rows, int b_cols, int a_cols) {
    bmm_blas_parallel(a, b, c, batch_dim, a_rows, b_cols, a_cols);
}


TEST_CASE("Timing and correctness of bmm optimized", "[bmm_opt]") {
    vector<tuple<int, int, int, int>> test_cases = {
        {10, 512, 512, 512},
        {5, 256, 256, 256},
        {10, 256, 512, 256},
        {15, 103, 90, 17},
        {8, 607, 398, 250}
    };

    for (const auto& [batch_dim, a_rows, b_cols, a_cols] : test_cases) {
        vector<double> a(batch_dim * a_rows * a_cols);
        vector<double> b(batch_dim * a_cols * b_cols);
        vector<double> c(batch_dim * a_rows * b_cols, 0.0);
        vector<double> c_ref(batch_dim * a_rows * b_cols, 0.0);

        generate_random_matrix(a, a_rows, a_cols);
        generate_random_matrix(b, a_cols, b_cols);

        // Reference implementation
        bmm_naive(a.data(), b.data(), c_ref.data(), batch_dim, a_rows, b_cols, a_cols);

        SECTION("Timing and correctness of bmm_blas with batch_dim=" + to_string(batch_dim) +
                ", a_rows=" + to_string(a_rows) + ", b_cols=" + to_string(b_cols) +
                ", a_cols=" + to_string(a_cols)) {
            fill(c.begin(), c.end(), 0.0);
            double time_taken = time_function(bmm_blas_wrapper, a.data(), b.data(), c.data(), batch_dim, a_rows, b_cols, a_cols);
            cout << "Time taken by bmm_blas with batch_dim=" << batch_dim << ", a_rows=" << a_rows
                 << ", b_cols=" << b_cols << ", a_cols=" << a_cols << ": " << time_taken << " seconds" << endl;
            REQUIRE(compare_matrices(c, c_ref));
        }
    }
}


