/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { return x > 0 ? x : 0; }

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    float *d_input;
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    float *d_h1a, *d_h2a, *d_out;
    float *d_delta1, *d_delta2, *d_delta3;

    cudaMalloc(&d_input, SIZE * sizeof(float));

    cudaMalloc(&d_W1, SIZE * H1 * sizeof(float)); cudaMalloc(&d_b1, H1 * sizeof(float));
    cudaMalloc(&d_W2, H1 * H2 * sizeof(float));   cudaMalloc(&d_b2, H2 * sizeof(float));
    cudaMalloc(&d_W3, H2 * CLASSES * sizeof(float)); cudaMalloc(&d_b3, CLASSES * sizeof(float));

    cudaMemcpy(d_W1, model->W1, SIZE * H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, model->b1, H1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W2, model->W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, model->b2, H2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W3, model->W3, H2 * CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, model->b3, CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_h1a, H1 * sizeof(float));
    cudaMalloc(&d_h2a, H2 * sizeof(float));
    cudaMalloc(&d_out, CLASSES * sizeof(float));

    cudaMalloc(&d_delta1, H1 * sizeof(float));
    cudaMalloc(&d_delta2, H2 * sizeof(float));
    cudaMalloc(&d_delta3, CLASSES * sizeof(float));

    float h1a[H1], h2a[H2], outa[CLASSES];
    float delta1[H1], delta2[H2], delta3[CLASSES];

    int threads = 128;
    int blocks_h1 = (H1 + threads - 1) / threads;
    int blocks_h2 = (H2 + threads - 1) / threads;
    int blocks_out = (CLASSES + threads - 1) / threads;

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ---------- Forward ----------
            cudaMemcpy(d_input, train_data[n], SIZE * sizeof(float), cudaMemcpyHostToDevice);

            matvec_forward<<<blocks_h1, threads>>>(d_W1, d_input, d_b1, d_h1a, SIZE, H1, true);
            cudaDeviceSynchronize();
            matvec_forward<<<blocks_h2, threads>>>(d_W2, d_h1a, d_b2, d_h2a, H1, H2, true);
            cudaDeviceSynchronize();
            matvec_forward<<<blocks_out, threads>>>(d_W3, d_h2a, d_b3, d_out, H2, CLASSES, false);
            cudaDeviceSynchronize();

            // Copy output back to host if you need it for loss calculation
            cudaMemcpy(outa, d_out, CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h1a, d_h1a, H1 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h2a, d_h2a, H2 * sizeof(float), cudaMemcpyDeviceToHost);

            softmax(outa, outa, CLASSES);

            // ---------- Loss ----------
            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa[k]+1e-8f);

            // ---------- Backprop ----------
            output_delta<<<blocks_out, threads>>>(d_delta3, d_out, d_train_label + n * CLASSES, CLASSES);
            cudaDeviceSynchronize();
            hidden_delta<<<blocks_h2, threads>>>(d_delta2, d_h2a, d_delta3, d_W3, H2, CLASSES);
            cudaDeviceSynchronize();
            hidden_delta<<<blocks_h1, threads>>>(d_delta1, d_h1a, d_delta2, d_W2, H1, H2);
            cudaDeviceSynchronize();


            // ---------- Update ----------
            int blocks_W1 = (SIZE * H1 + threads - 1) / threads;
            int blocks_W2 = (H1 * H2 + threads - 1) / threads;
            int blocks_W3 = (H2 * CLASSES + threads - 1) / threads;

            int blocks_b1 = (H1 + threads - 1) / threads;
            int blocks_b2 = (H2 + threads - 1) / threads;
            int blocks_b3 = (CLASSES + threads - 1) / threads;
            update_weights<<<blocks_W1, threads>>>(d_W1, d_delta1, d_input, SIZE, H1, LR);
            update_biases<<<blocks_b1, threads>>>(d_b1, d_delta1, H1, LR);

            update_weights<<<blocks_W2, threads>>>(d_W2, d_delta2, d_h1a, H1, H2, LR);
            update_biases<<<blocks_b2, threads>>>(d_b2, d_delta2, H2, LR);

            update_weights<<<blocks_W3, threads>>>(d_W3, d_delta3, d_h2a, H2, CLASSES, LR);
            update_biases<<<blocks_b3, threads>>>(d_b3, d_delta3, CLASSES, LR);
            cudaDeviceSynchronize();
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }
    cudaMemcpy(d_W1, model->W1, SIZE * H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, model->b1, H1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, model->W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, model->b2, H2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, model->W3, H2 * CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, model->b3, CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // ---------- Free device memory ----------
    cudaFree(d_input);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_h1a); cudaFree(d_h2a); cudaFree(d_out);
    cudaFree(d_delta1); cudaFree(d_delta2); cudaFree(d_delta3);
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


