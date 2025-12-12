#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); } \
        } while(0)

// =================================================================================
// 1. Reduce Kernel (Sum Reduction)
// =================================================================================
template <typename T>
__global__ void reduce_sum_kernel(const T* __restrict__ x, T* __restrict__ y, int num_elements) {
    extern __shared__ T sdata[]; 
    int tid = threadIdx.x;
    int row = blockIdx.x;        // 每个 block 处理一行
    T sum = 0;
    
    // 每个线程对该行的多个元素求部分和; 跨步读取
    for (int i = tid; i < num_elements; i += blockDim.x) {
        sum += x[row * num_elements + i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        y[row] = sdata[0];
    }
}


// =================================================================================
// 2. Softmax Kernel (Row-wise)
// =================================================================================



// =================================================================================
// 3. GEMV Kernel (y = A * x)
// =================================================================================


// =================================================================================
// 4. GEMM Kernel (C = A * B)
// =================================================================================





// =================================================================================
// Main / Test
// =================================================================================
int main() {
    // --- Test Reduce ---
    {
        std::cout << "\n--- Testing Reduce (Sum) ---" << std::endl;
        int batch = 10;           // 批次数量
        int num_elements = 4096;   // 每行元素数
        int threadsPerBlock = 256; 
        dim3 grid(batch);          
        dim3 block(threadsPerBlock);

        // Host buffers
        std::vector<float> h_x(batch * num_elements,1.0f);
        std::vector<float> h_y(batch, 0.0f);

        // Device buffers
        float* d_x = nullptr;
        float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * batch * num_elements));
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * batch));

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * batch * num_elements, cudaMemcpyHostToDevice));

        // Launch kernel with dynamic shared memory: threadsPerBlock * sizeof(float)
        reduce_sum_kernel<float><<<grid, block, threadsPerBlock * sizeof(float)>>>(d_x, d_y, num_elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, sizeof(float) * batch, cudaMemcpyDeviceToHost));

        // Verify and print
        for (int r = 0; r < batch; ++r) {
            std::cout << h_y[r]  << std::endl;
        }

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    // // --- Test Softmax ---
    // {

    // }

    // // --- Test GEMV ---
    // {

    // }

    // // --- Test GEMM ---
    // {

    // }

    return 0;
}