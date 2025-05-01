#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>

#define INT_BUFFER_SIZE 12
#define RANDOM_SEED 1234ULL // Seed for random number generation, ULL stands for unsigned long long
#define __cuRAND_DEVICEWIDE_RNG__ false
#define __cuRAND_KERNEL_RNG__ true
#define NTHREADS 5

#define SPHERE_RADIUS 1.0f
#define SPHERE_CENTER_X 0.0f
#define SPHERE_CENTER_Y 0.0f

__device__ void intToString(int value, char* buffer) {
    char temp[INT_BUFFER_SIZE]; // Enough for a 32-bit int including sign and '\0'
    int i = 0;
    bool isNegative = false;

    if (value < 0) {
        isNegative = true;
        value = -value;
    }

    // Extract digits in reverse order
    do {
        temp[i++] = (value % 10) + '0';
        value /= 10;
    } while (value > 0);

    if (isNegative) {
        temp[i++] = '-';
    }
    
    // Reverse the digits into the output buffer
    int j = 0;
    while(i > 0) {
        buffer[j++] = temp[--i];
    }
    buffer[j] = '\0';
}

__device__ void intToString(unsigned int value, char* buffer) {
    char temp[INT_BUFFER_SIZE];
    int i = 0;
    do {
        temp[i++] = (value % 10) + '0';
        value /= 10;
    } while (value > 0);
    
    int j = 0;
    while (i > 0) {
        buffer[j++] = temp[--i];
    }
    buffer[j] = '\0';
}

__global__ void welcome(char* msg) {
    // Simple hello world kernel. printf writes to host output stream directly.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("My thread idx is %d\n", idx);
}

__global__ void write_hello_to_string(int msg_len, int msg_len_thread, char* msg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * msg_len_thread < msg_len){
        char* threadmsg = &msg[idx * msg_len_thread];
        char intBuffer[INT_BUFFER_SIZE]; // Buffer to hold the integer as a string
        intToString(idx, intBuffer); // Convert thread index to string

        for (int i = 0; i < INT_BUFFER_SIZE; i++) {
            threadmsg[i] = intBuffer[i];
        }
        threadmsg[INT_BUFFER_SIZE] = '\0'; // Null-terminate the string
    }
}

__global__ void write_random_to_string(curandState* state, int msg_len, int msg_len_thread, char* msg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * msg_len_thread < msg_len){
        char* threadmsg = &msg[idx * msg_len_thread];
        char intBuffer[INT_BUFFER_SIZE]; // Buffer to hold the integer as a string
        uint random_int = curand(&state[idx]);

        intToString(random_int, intBuffer); // Convert thread index to string

        for (int i = 0; i < INT_BUFFER_SIZE; i++) {
            threadmsg[i] = intBuffer[i];
        }
        threadmsg[INT_BUFFER_SIZE] = '\0'; // Null-terminate the string
    }
}

__global__ void setupkernel(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

int main() {



    std::cout << "Hello from CUDA" << "\n";
    const int nThreads = NTHREADS;

    const int messageSizePerThread = 50;
    char* host_messages = (char*) malloc(nThreads * (messageSizePerThread + 1) * sizeof(char));
    char* d_msg;
    int messageSize = nThreads * (messageSizePerThread + 1) * sizeof(char);

    // Allocate device memory
    cudaMalloc((void**) &d_msg, messageSize);
    // Copy message to constant memory
    cudaMemcpy(host_messages, d_msg, messageSize, cudaMemcpyHostToDevice);
    // Launch welcome kernel
    write_hello_to_string<<<1, 10*nThreads>>>(messageSize, messageSizePerThread+1, d_msg); 
    // Copy result back to host
    cudaMemcpy(host_messages, d_msg, messageSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nThreads; i++) {
        std::cout << "Thread " << i << ": " << host_messages + i * (messageSizePerThread + 1) << "\n";
    }
    cudaMemset(host_messages, 0, messageSize);

    #if __cuRAND_DEVICEWIDE_RNG__
    printf("cuRAND device-wide RNG is enabled.\n");
    float* d_random_numbers;
    float* h_random_numbers;

    h_random_numbers = (float*) malloc(NThreads * sizeof(float));
    cudaMalloc((void**) &d_random_numbers, NThreads * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, RANDOM_SEED);
    curandGenerateUniform(gen, d_random_numbers, NThreads);
    cudaMemcpy(h_random_numbers, d_random_numbers, NThreads * sizeof(float), cudaMemcpyDeviceToHost);

    /* Show result RNG back on host*/
    for(int i = 0; i < NThreads; i++) {
        printf("Random number %d: %1.4f ", i,h_random_numbers[i]);
    }
    printf("\n");
    curandDestroyGenerator(gen);
    cudaFree(d_random_numbers);
    free(h_random_numbers);
    #endif

    #if __cuRAND_KERNEL_RNG__
    printf("cuRAND kernel RNG is enabled.\n");
    float* d_random_numbers;
    float* h_random_numbers;

    h_random_numbers = (float*) malloc(nThreads * sizeof(float));
    cudaMalloc((void**) &d_random_numbers, nThreads * sizeof(float));

    curandState* devStates;
    cudaMalloc((void**) &devStates, nThreads * sizeof(curandState));
    setupkernel<<<1, nThreads>>>(devStates, RANDOM_SEED);
    cudaDeviceSynchronize();
    for (int iter = 0; iter < 3; iter++){
        write_random_to_string<<<1, nThreads>>>(devStates, messageSize, messageSizePerThread+1, d_msg);
        cudaMemcpy(host_messages, d_msg, messageSize, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < nThreads; i++) {
            std::cout << "Thread RNG" << i << ": " << host_messages + i * (messageSizePerThread + 1) << "\n";
        }
        cudaMemset(host_messages, 0, messageSize);
    }

    // Free device memory
    cudaFree(devStates);
    cudaFree(d_random_numbers);
    free(h_random_numbers);
    #endif
    


    //std::cout << h_msg << "\n";
    
    // Cleanup
    free(host_messages);
    cudaFree(d_msg);
    
    return 0;
}