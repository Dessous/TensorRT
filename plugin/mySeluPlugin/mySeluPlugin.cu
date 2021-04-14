/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "mySeluPlugin.h"
#include <cuda_fp16.h>


    __global__ void kernelSelu(
        int N,
        int iH,
        int iW,
        float alpha,
        float lambda,
        float* outputs
        )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N){
        outputs[index] = outputs[index] >= 0 ? outputs[index] * lambda : alpha * lambda * (expf(outputs[index]) - 1);
    }
    __syncthreads();
}

int inferenceSelu(
    int batchSize,
    int iC,
    int iH,
    int iW,
    float alpha,
    float lambda,
    float* inputs,
    float* outputs,
    cudaStream_t stream){
        // NCHW
        const int nThreads = 512;
        int len = iC * iH * iW;

        int nBlocks = (int)((float)len / nThreads) + 1;

        for(int i=0; i<batchSize; ++i){
            // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
            cudaMemcpy(outputs, inputs, sizeof(float) * len, cudaMemcpyDeviceToDevice);
            kernelSelu<<<nBlocks, nThreads, 0, stream>>>(len, iH, iW, alpha, lambda, outputs);
            outputs += len;
            inputs += len;
        }

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                __FILE__, __LINE__, cudaGetErrorString( err ) );
        return 1;
    }
    return 0;
}

int mySeluPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    return inferenceSelu(batchSize, iC, iH, iW, alpha, lambda, (float*)inputs[0], (float*)outputs[0], stream);
}
