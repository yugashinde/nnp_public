/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

__global__ void matvec_forward( 
    float* mat,      // weights: rows = input size, cols = output size
    float* vec_in,   // input vector
    float* bias,     // biases for each output neuron
    float* vec_out,  // output vector
    int input_size,
    int output_size,
    bool apply_relu);


__global__ void output_delta(
    float* delta3,        // output deltas
    const float* outa,    // softmax outputs
    const float* label,   // true labels
    int num_classes
);

__global__ void hidden_delta(
    float* delta_hidden,       // delta for current hidden layer
    const float* act_hidden,   // activation values of current layer (ReLU)
    const float* delta_next,   // delta from next layer
    const float* W_next,       // weights connecting current layer to next layer
    int num_hidden,            // number of neurons in current layer
    int num_next               // number of neurons in next layer
);

__global__ void update_weights(
    float* W,          // weight matrix: rows = input_size, cols = output_size
    const float* delta, // delta vector for this layer (output_size)
    const float* act_in, // input activations from previous layer (input_size)
    int input_size,
    int output_size,
    float lr);

__global__ void update_biases(
    float* b,          // bias vector
    const float* delta,// delta vector for this layer
    int size,
    float lr) ;      // learning rate
