#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include "learn_kernels.cuh"
#include "cudamat.cuh"

extern "C" {

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

extern int mult_by_sigmoid_deriv(cudamat* target, cudamat* acts) {
    int len = acts->size[0]*acts->size[1];

    if (acts->is_trans != target->is_trans)
        return ERROR_TRANSPOSED;

    if (acts->size[0] != target->size[0] || acts->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultiplyBySigmoidGrad<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(acts->data_device, target->data_device, len);

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

extern int calculate_l1_penalty(cudamat* mat, float alpha, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCalculateL1Penalty<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(mat->data_device, alpha, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

}
