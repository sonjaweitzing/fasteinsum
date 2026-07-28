#include "catch.hpp"
#include "bmm.h"
#include <vector>

std::vector<double> flatten(const std::vector<std::vector<int>>& matrix) {
    std::vector<double> flat;
    for (const auto& row : matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

std::vector<double> flatten(const std::vector<std::vector<std::vector<int>>>& tensor) {
    std::vector<double> flat;
    for (const auto& matrix : tensor) {
        for (const auto& row : matrix) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    return flat;
}

std::vector<std::vector<int>> unflatten(const std::vector<double>& flat, int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<int>(flat[i * cols + j]);
        }
    }
    return matrix;
}

std::vector<std::vector<std::vector<int>>> unflatten(const std::vector<double>& flat, int batch, int rows, int cols) {
    std::vector<std::vector<std::vector<int>>> tensor(batch, std::vector<std::vector<int>>(rows, std::vector<int>(cols)));
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                tensor[b][i][j] = static_cast<int>(flat[b * rows * cols + i * cols + j]);
            }
        }
    }
    return tensor;
}

TEST_CASE("bmm function - easy cases", "[bmm]") {
    SECTION("Multiplying two 2x2 matrices") {
        std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
        std::vector<std::vector<int>> B = {{5, 6}, {7, 8}};
        std::vector<std::vector<int>> expected = {{19, 22}, {43, 50}};

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(4);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 1, 2, 2, 2);

        REQUIRE(unflatten(result, 2, 2) == expected);
    }

    SECTION("Multiplying two 3x3 matrices") {
        std::vector<std::vector<int>> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::vector<std::vector<int>> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        std::vector<std::vector<int>> expected = {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}};

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(9);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 1, 3, 3, 3);

        REQUIRE(unflatten(result, 3, 3) == expected);
    }

    SECTION("Multiplying identity matrix with another matrix") {
        std::vector<std::vector<int>> A = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        std::vector<std::vector<int>> B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::vector<std::vector<int>> expected = B;

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(9);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 1, 3, 3, 3);

        REQUIRE(unflatten(result, 3, 3) == expected);
    }

    SECTION("Multiplying zero matrix with another matrix") {
        std::vector<std::vector<int>> A = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        std::vector<std::vector<int>> B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::vector<std::vector<int>> expected = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(9);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 1, 3, 3, 3);

        REQUIRE(unflatten(result, 3, 3) == expected);
    }

    SECTION("Multiplying two 3D tensors with batch size 2") {
        std::vector<std::vector<std::vector<int>>> A = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}}
        };
        std::vector<std::vector<std::vector<int>>> B = {
            {{9, 10}, {11, 12}},
            {{13, 14}, {15, 16}}
        };
        std::vector<std::vector<std::vector<int>>> expected = {
            {{31, 34}, {71, 78}},
            {{155, 166}, {211, 226}}
        };

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(8);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 2, 2, 2, 2);

        REQUIRE(unflatten(result, 2, 2, 2) == expected);
    }

    SECTION("Multiplying two 3D tensors with batch size 3") {
        std::vector<std::vector<std::vector<int>>> A = {
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
            {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}}
        };
        std::vector<std::vector<std::vector<int>>> B = {
            {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}},
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
        };
        std::vector<std::vector<std::vector<int>>> expected = {
            {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}},
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
            {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}}
        };

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(27);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 3, 3, 3, 3);

        REQUIRE(unflatten(result, 3, 3, 3) == expected);
    }

    SECTION("Multiplying 3D tensor with identity matrices in batch") {
        std::vector<std::vector<std::vector<int>>> A = {
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
            {{2, 3, 4}, {5, 6, 7}, {8, 9, 10}}
        };
        std::vector<std::vector<std::vector<int>>> B = {
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
            {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
        };
        std::vector<std::vector<std::vector<int>>> expected = A;

        auto flat_A = flatten(A);
        auto flat_B = flatten(B);
        std::vector<double> result(18);

        bmm_naive(flat_A.data(), flat_B.data(), result.data(), 2, 3, 3, 3);

        REQUIRE(unflatten(result, 2, 3, 3) == expected);
    }
}