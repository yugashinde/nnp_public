/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

#include "kernels.h"

__global__ void matvec_forward(
    float* mat,      // weights: rows = input size, cols = output size
    float* vec_in,   // input vector
    float* bias,     // biases for each output neuron
    float* vec_out,  // output vector
    int input_size,
    int output_size,
    bool apply_relu) // flag to apply ReLU
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= output_size) return;

    float sum = bias[idx];
    for (int i = 0; i < input_size; i++) {
        sum += mat[i * output_size + idx] * vec_in[i];
    }

    if (apply_relu) {
        vec_out[idx] = sum > 0 ? sum : 0; // ReLU
    } else {
        vec_out[idx] = sum; // output layer
    }
}

__global__ void output_delta(
    float* delta3,        // output deltas
    const float* outa,    // softmax outputs
    const float* label,   // true labels
    int num_classes
){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= num_classes) return;

    delta3[k] = label[k] - outa[k]; // delta = y_true - y_pred
}

__global__ void hidden_delta(
    float* delta_hidden,       // delta for current hidden layer
    const float* act_hidden,   // activation values of current layer (ReLU)
    const float* delta_next,   // delta from next layer
    const float* W_next,       // weights connecting current layer to next layer
    int num_hidden,            // number of neurons in current layer
    int num_next               // number of neurons in next layer
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j >= num_hidden) return;

    float err = 0.0f;
    for(int k = 0; k < num_next; k++)
        err += delta_next[k] * W_next[j * num_next + k]; // sum of weighted deltas

    // Multiply by ReLU derivative
    delta_hidden[j] = err * (act_hidden[j] > 0 ? 1.0f : 0.0f);
}