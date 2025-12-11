#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstddef>

namespace op {

void reduce_sum(const float* input, float* output, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    *output = sum;
}

void softmax(const float* input, float* output, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        const float* row_in = input + i * cols;
        float* row_out = output + i * cols;

        // Find max for numerical stability
        float max_val = row_in[0];
        for (size_t j = 1; j < cols; ++j) {
            if (row_in[j] > max_val) {
                max_val = row_in[j];
            }
        }

        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float val = std::exp(row_in[j] - max_val);
            row_out[j] = val;
            sum_exp += val;
        }

        // Normalize
        for (size_t j = 0; j < cols; ++j) {
            row_out[j] /= sum_exp;
        }
    }
}

void gemm(const float* A, const float* B, float* C,
          size_t M, size_t N, size_t K,
          float alpha, float beta,
          bool transA, bool transB) {
    
    // C is MxN
    // If transA=false, A is MxK. A[i, k] = A[i*K + k]
    // If transA=true,  A is KxM. A[i, k] (logical) = A[k*M + i] (physical)
    
    // If transB=false, B is KxN. B[k, j] = B[k*N + j]
    // If transB=true,  B is NxK. B[k, j] (logical) = B[j*K + k] (physical)

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_val = transA ? A[k * M + i] : A[i * K + k];
                float b_val = transB ? B[j * K + k] : B[k * N + j];
                sum += a_val * b_val;
            }
            // C = alpha * (A*B) + beta * C
            // Note: usually beta applies to the existing value in C
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void gemv(const float* A, const float* x, float* y,
          size_t M, size_t N,
          float alpha, float beta,
          bool transA) {
    
    // A is MxN physically.
    // If transA=false: y = alpha * A * x + beta * y
    // y is M-dim, x is N-dim.
    // y[i] = alpha * sum(A[i, j] * x[j]) + beta * y[i]
    
    // If transA=true: y = alpha * A^T * x + beta * y
    // y is N-dim, x is M-dim.
    // y[j] = alpha * sum(A[i, j] * x[i]) + beta * y[j]

    if (!transA) {
        // Output is vector of size M
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        // Output is vector of size N
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < M; ++i) {
                // A^T[j, i] corresponds to A[i, j]
                sum += A[i * N + j] * x[i];
            }
            y[j] = alpha * sum + beta * y[j];
        }
    }
}

} // namespace op

// Helper to check approximate equality
bool is_close(float a, float b, float tol = 1e-5) {
    return std::abs(a - b) < tol;
}

void test_reduce_sum() {
    std::cout << "Testing Reduce Sum..." << std::endl;
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
    float output = 0.0f;
    op::reduce_sum(input.data(), &output, input.size());
    assert(is_close(output, 10.0f));
    std::cout << "Passed!" << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax..." << std::endl;
    // Test 1x3 input: [1.0, 2.0, 3.0]
    // exp: [2.718, 7.389, 20.085], sum = 30.192
    // softmax: [0.090, 0.244, 0.665]
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3);
    op::softmax(input.data(), output.data(), 1, 3);
    
    assert(is_close(output[0], 0.09003f));
    assert(is_close(output[1], 0.24473f));
    assert(is_close(output[2], 0.66524f));
    
    // Check sum is 1
    float sum = 0.0f;
    for(float v : output) sum += v;
    assert(is_close(sum, 1.0f));
    
    std::cout << "Passed!" << std::endl;
}

void test_gemm() {
    std::cout << "Testing GEMM..." << std::endl;
    // C = 1.0 * A * B + 0.0 * C
    // A (2x3):
    // 1 2 3
    // 4 5 6
    // B (3x2):
    // 7 8
    // 9 10
    // 11 12
    // Expected C (2x2):
    // 1*7+2*9+3*11  1*8+2*10+3*12  => 7+18+33=58   8+20+36=64
    // 4*7+5*9+6*11  4*8+5*10+6*12  => 28+45+66=139 32+50+72=154
    
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    std::vector<float> C(4, 0.0f);
    
    op::gemm(A.data(), B.data(), C.data(), 2, 2, 3, 1.0f, 0.0f, false, false);
    
    assert(is_close(C[0], 58.0f));
    assert(is_close(C[1], 64.0f));
    assert(is_close(C[2], 139.0f));
    assert(is_close(C[3], 154.0f));

    std::cout << "Passed!" << std::endl;
}

void test_gemv() {
    std::cout << "Testing GEMV..." << std::endl;
    // y = 1.0 * A * x + 1.0 * y
    // A (2x3):
    // 1 2 3
    // 4 5 6
    // x (3):
    // 1
    // 1
    // 1
    // y_in (2): 1, 1
    // Expected y_out:
    // (1+2+3) + 1 = 7
    // (4+5+6) + 1 = 16
    
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> x = {1, 1, 1};
    std::vector<float> y = {1, 1};
    
    op::gemv(A.data(), x.data(), y.data(), 2, 3, 1.0f, 1.0f, false);
    
    assert(is_close(y[0], 7.0f));
    assert(is_close(y[1], 16.0f));
    
    // Test Transposed A
    // y = A^T * x
    // A (2x3), A^T (3x2)
    // 1 4
    // 2 5
    // 3 6
    // x (2): 1, 1
    // y_in (3): 0, 0, 0
    // Expected y_out:
    // 1*1 + 4*1 = 5
    // 2*1 + 5*1 = 7
    // 3*1 + 6*1 = 9
    
    std::vector<float> x_t = {1, 1};
    std::vector<float> y_t = {0, 0, 0};
    
    op::gemv(A.data(), x_t.data(), y_t.data(), 2, 3, 1.0f, 0.0f, true);
    
    assert(is_close(y_t[0], 5.0f));
    assert(is_close(y_t[1], 7.0f));
    assert(is_close(y_t[2], 9.0f));
    
    std::cout << "Passed!" << std::endl;
}

int main() {
    test_reduce_sum();
    test_softmax();
    test_gemm();
    test_gemv();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}