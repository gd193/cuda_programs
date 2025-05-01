#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>

#define INT_BUFFER_SIZE 12
#define RANDOM_SEED 1234ULL // Seed for random number generation, ULL stands for unsigned long long
#define NPOINTS 1000
#define NWALKS 1000
#define MAXSTEPS 1000

#define SPHERE_RADIUS 1.0f
#define SPHERE_CENTER_X 0.0f
#define SPHERE_CENTER_Y 0.0f

#define __TIME_KERNEL__ true
#define __PRINT_DEVICE_INFO__ true
#define __PRINT_RESULTS__ flase
#define __CUDA_DEVICE_NR__ 1


__device__ void sample_2d_sphere_surface_around_point(curandState* state, float* surfaceX, float* surfaceY, float centerX, float centerY, float radius){
    float theta = curand_uniform(state) * 2.0 * CUDART_PI_F;
    float x = radius * cosf(theta) + centerX;
    float y = radius * sinf(theta) + centerY;
    *surfaceX = x;
    *surfaceY = y; 
}

__device__ void sdf_sphere(float* distance, float pointX, float pointY, float centerX, float centerY, float radius){
    float dx = pointX - centerX;
    float dy = pointY - centerY;
    *distance = sqrtf(dx * dx + dy * dy) - radius;
}

__device__ void boundary_condition_test(float* boundaryValue, float pointX, float pointY) {
    *boundaryValue = 1.0f + 2.0f * sinf(atanf(pointY / pointX));
}

__global__ void WoS_kernel(curandState* state, float* solutionEstimate, bool* termininated, float* pointX, float* pointY, int nPoints, int maxSteps, float eps) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nPoints) {
        float x = pointX[idx];
        float y = pointY[idx];
        float distanceToBoundary = 10.0f * eps;

        for (int step = 0; step < maxSteps; step++) {

            sdf_sphere(&distanceToBoundary, x, y, SPHERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_RADIUS);
            if (abs(distanceToBoundary) < eps) {
                boundary_condition_test(&solutionEstimate[idx], x, y);
                termininated[idx] = true;
                break;
            }

            // Sample a point on the surface of a sphere
            sample_2d_sphere_surface_around_point(&state[idx], &x, &y, x, y, distanceToBoundary);
        
        }
    }
    return;
}

__global__ void setupRNG(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__host__ void sample_in_2d_ball(std::mt19937& gen, float* pointsX, float* pointsY, int nPoints, float radius){
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < nPoints; i++){
        float theta = dist(gen) * 2.0 * CUDART_PI_F;
        float r = dist(gen) * sqrtf(radius);

        pointsX[i] = r * cosf(theta);
        pointsY[i] = r * sinf(theta);
    }
}

__host__ void repeat_each_element_n_times(float* input, float* output, int nElements, int nRepeats) {
    // assuming intput has nElements and output has nElements * nRepeats allocated
    float *buffer;
    buffer = (float*)malloc(nElements * sizeof(float));
    memcpy(buffer, input, nElements * sizeof(float));

    for (int i=0; i<nElements; i++){
        for (int j=0; j<nRepeats; j++){
            output[i*nRepeats + j] = buffer[i]; //each element is repeated nRepeats times next to each other xyz --> xx...xy....yz...z
        }
    }
    free(buffer);
}

int main() {
    cudaSetDevice(__CUDA_DEVICE_NR__);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, __CUDA_DEVICE_NR__);
    std::cout << "Hello from CUDA" << "\n";

    #if __PRINT_DEVICE_INFO__
    std::cout << "CUDA device properties:\n";
    std::cout << "Using device: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Max shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    #endif

    const int nPoints = NPOINTS;
    const int nWalks = NWALKS;
    const float eps = 0.001f;


    // Allocate memory for points and solution estimates
    float *h_pointsX, *h_pointsY, *h_solutionEstimate;
    bool *h_isTerminated;
    h_pointsX = (float*)malloc(nPoints * nWalks * sizeof(float));
    h_pointsY = (float*)malloc(nPoints * nWalks * sizeof(float));
    h_solutionEstimate = (float*)malloc(nPoints * nWalks * sizeof(float));
    memset(h_solutionEstimate, 0, nPoints * nWalks * sizeof(float));
    h_isTerminated = (bool*)malloc(nPoints * nWalks * sizeof(bool));
    memset(h_isTerminated, 0, nPoints * nWalks * sizeof(bool));

    float *d_pointsX, *d_pointsY, *d_solutionEstimate;
    bool *d_isTerminated;
    cudaMalloc((void**)&d_pointsX, nPoints * nWalks * sizeof(float));
    cudaMalloc((void**)&d_pointsY, nPoints * nWalks * sizeof(float));
    cudaMalloc((void**)&d_solutionEstimate, nPoints * nWalks * sizeof(float));
    cudaMalloc((void**)&d_isTerminated, nPoints * nWalks * sizeof(bool));

    // calculate the block and threads for the kernel
    int nThreads = min(nPoints * nWalks, 1024);
    int nBlocks = (nPoints * nWalks + nThreads - 1) / nThreads;

    // Init random number generator
    curandState* d_states;
    cudaMalloc((void**)&d_states, nPoints * nWalks * sizeof(curandState));
    setupRNG<<<nBlocks, nThreads>>>(d_states, RANDOM_SEED);
    cudaDeviceSynchronize();

    // Initialize points
    std::mt19937 gen(RANDOM_SEED);
    
    sample_in_2d_ball(gen, h_pointsX, h_pointsY, nPoints, SPHERE_RADIUS);
    repeat_each_element_n_times(h_pointsX, h_pointsX, nPoints, nWalks);
    repeat_each_element_n_times(h_pointsY, h_pointsY, nPoints, nWalks);

    // Copy points to device
    cudaMemcpy(d_pointsX, h_pointsX, nPoints * nWalks * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pointsY, h_pointsY, nPoints * nWalks * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solutionEstimate, h_solutionEstimate, nPoints * nWalks * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_isTerminated, h_isTerminated, nPoints * nWalks * sizeof(bool), cudaMemcpyHostToDevice);

    #if __TIME_KERNEL__
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    WoS_kernel<<<nBlocks, nThreads>>>(d_states, d_solutionEstimate, d_isTerminated, d_pointsX, d_pointsY, nPoints*nWalks, MAXSTEPS, eps);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time CUDA WoS kernel (without data setup/transfer): " << elapsed.count() << " milliseconds\n";
    #else
    WoS_kernel<<<nBlocks, nThreads>>>(d_states, d_solutionEstimate, d_isTerminated, d_pointsX, d_pointsY, nPoints*nWalks, MAXSTEPS, eps);
    cudaDeviceSynchronize()
    #endif
    // Copy results back to host
    cudaMemcpy(h_solutionEstimate, d_solutionEstimate, nPoints * nWalks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_isTerminated, d_isTerminated, nPoints * nWalks * sizeof(bool), cudaMemcpyDeviceToHost);

    #if __PRINT_RESULTS__
    // Print results
    int nPointsPrint = min(nPoints, 10);
    int nWalksPrint = min(nWalks, 10);
    for (int i = 0; i < nPointsPrint; i++) {
        std::cout << "Point " << i << " at position (" << h_pointsX[i*nWalks] << ", " << h_pointsY[i*nWalks] << "). \n";
        for (int j = 0; j < nWalksPrint; j++) {
            std::cout << "\t Walk " << j << ": Terminated = " << h_isTerminated[i*nWalks + j] << ", Solution Estimate = " << h_solutionEstimate[i*nWalks + j] << "\n";
        }
    }
    #endif

    // Free device memory
    cudaFree(d_pointsX);
    cudaFree(d_pointsY);
    cudaFree(d_solutionEstimate);
    cudaFree(d_isTerminated);
    cudaFree(d_states);

    // Free host memory
    free(h_pointsX);
    free(h_pointsY);
    free(h_solutionEstimate);
    free(h_isTerminated);

    return 0;
}