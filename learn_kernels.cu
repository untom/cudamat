#include "learn_kernels.cuh"

__global__ void kMultiplyBySigmoidGrad(float* act, float* target, const unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for(unsigned int i = idx; i < len; i+= numThreads) {
        target[i] = target[i] * act[i] * (1.0f - act[i]);
    }
}


__global__ void kCalculateL1Penalty(float* mat, float alpha, float* target, const unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    float s;
    float d;
    
    for(unsigned int i = idx; i < len; i+= numThreads) {
        s = mat[i] ? copysignf(1., mat[i]) : 0.0;
        target[i] = alpha > fabsf(mat[i]) ? mat[i] : alpha*s;
    }
}
