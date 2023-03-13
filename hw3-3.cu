#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <omp.h>
#define B 64
#define HALF 32
// #define DEV_NO 0


using namespace std;
const int INF = ((1 << 30) - 1);
__global__ void phase1(int *dst, int Round, int N);
// __global__ void phase2_hor(int* d_dist, int Round, int n);
// __global__ void phase2_ver(int* d_dist, int Round, int n);
__global__ void phase2(int* d_dist, int Round, int n);
__global__ void phase3(int* d_dist, int Round, int n, int start);
// __global__ void bound(int* d_dist, int n, int start);

// int n, m;
int main(int argc, char* argv[]) {
    int n, m;
    int *Dist;
    int *d_dist[2];   
    // int ** d_dist = NULL;
    FILE* file = fopen(argv[1], "rb"); 
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    int padding_n = int((n + B - 1) / B) * B;
    long int size = padding_n * padding_n * sizeof(int);


    // pin memory
    Dist = (int *)malloc(size);
    cudaHostRegister(Dist, size, cudaHostRegisterDefault);
    // printf("%lu\n", size);
    // cudaMallocHost(&Dist, size);
    

    int *buf = (int*)malloc(m*3*sizeof(int));
    fread(buf, sizeof(int), m * 3, file);
    #pragma unroll 32
    // #pragma omg parallel for collapse(2)
    for (int i = 0; i < padding_n; ++i) {
        // printf("%d\n", i);
        for (int j = 0; j < padding_n; ++j) {
            // printf("%d\n", j);
            if (i == j) {
                Dist[i * padding_n + j] = 0;
            } else {
                Dist[i * padding_n + j] = INF;
            }
        }
    }
    // cudaHostRegister(Dist, size, cudaHostRegisterDefault);
    for (int i = 0; i < m; ++i) {
        Dist[buf[i * 3]*padding_n + buf[i * 3 + 1]] = buf[i*3 + 2];
    }
    fclose(file);
    
    int round = padding_n / B;
    
    dim3 blockDim(HALF, HALF);
    
    
    #pragma omp parallel num_threads(2)
    {
        
        int CPU_ID = omp_get_thread_num();
        cudaSetDevice(CPU_ID);
        cudaMalloc(&d_dist[CPU_ID], size);
        int round_c = (round / 2) * (1-CPU_ID) + ((round + 1) / 2) * CPU_ID;
        
        // int startDist = start * padding_n * B;
        long int sizeGPU = ((round / 2) * (1-CPU_ID) + round_c * CPU_ID) * B * padding_n * sizeof(int);
        long int row = padding_n * B;
        int start = round / 2 * CPU_ID;
        long int startDist = start * row;
        
        cudaMemcpyAsync(d_dist[CPU_ID] + startDist, Dist + startDist, sizeGPU, cudaMemcpyHostToDevice);
        
        dim3 gridDim1(round, round_c);
        // dim3 gridDim2(round, round_c);
        cudaDeviceEnablePeerAccess(1 - CPU_ID, 0);
        // printf("%d %d %d , start = %d, total_row = %d, startDist=%d\n", padding_n, round, CPU_ID, start, round_c, startDist);
        #pragma omp barrier
        for(int r = 0; r < round; ++r){
            if(r >= start && r < (round_c + start)){
                // printf("%d, ID = %d\n", r, CPU_ID);
                cudaMemcpy(d_dist[1-CPU_ID] + (row * r), d_dist[CPU_ID] + (row * r), row * sizeof(int), cudaMemcpyDeviceToDevice);
                // printf("%d from %d, startRow = %d, round %d\n", 1-CPU_ID, CPU_ID, sizeGPU, r);
            }
            #pragma omp barrier
            phase1 <<<1, blockDim>>>(d_dist[CPU_ID], r, padding_n);
            phase2 <<<round, blockDim>>>(d_dist[CPU_ID], r, padding_n);
            phase3 <<<gridDim1, blockDim>>>(d_dist[CPU_ID], r, padding_n, start);
        }
        
        cudaMemcpy(Dist + startDist, d_dist[CPU_ID] + startDist, sizeGPU, cudaMemcpyDeviceToHost);
        cudaFree(d_dist[CPU_ID]);
    }

    FILE* outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i * padding_n + j] >= INF) Dist[i * padding_n + j] = INF;
        }
        fwrite(&Dist[i * padding_n], sizeof(int), n, outfile);
    }
    // fclose(outfile);
    free(Dist);
    // cudaFree(d_dist[0]);
    // cudaFree(d_dist[1]);
    
    return 0;
}

__global__ void phase1(int* d_dist, int Round, int n){
    int i = threadIdx.y;
    int j = threadIdx.x;

    // if(i >= n || j >= n) return;

    __shared__ int temp_d[B][B];

    temp_d[i][j] = d_dist[(Round * B) * (n + 1) + i * n + j];
    temp_d[i + HALF][j] = d_dist[(Round * B) * (n + 1) + (i + HALF) * n + j];
    temp_d[i][j + HALF] = d_dist[(Round * B) * (n + 1) + i * n + j + HALF];
    temp_d[i + HALF][j + HALF] = d_dist[(Round * B + HALF) * (n + 1) + i * n + j];
    __syncthreads();

    #pragma unroll 32
    for(int k = 0; k < B; ++k){
        temp_d[i][j] = min(temp_d[i][k] + temp_d[k][j], temp_d[i][j]);
        temp_d[i+HALF][j] = min(temp_d[i+HALF][k] + temp_d[k][j], temp_d[i+HALF][j]);
        temp_d[i][j+HALF] = min(temp_d[i][k] + temp_d[k][j+HALF], temp_d[i][j+HALF]);
        temp_d[i+HALF][j+HALF] = min(temp_d[i+HALF][k] + temp_d[k][j+HALF], temp_d[i+HALF][j+HALF]);
        __syncthreads();
    }

    //copy to shared memory
    d_dist[(Round * B) * (n + 1) + i * n + j] = temp_d[i][j];
    d_dist[(Round * B) * (n + 1) + (i + HALF) * n + j] = temp_d[i + HALF][j];
    d_dist[(Round * B) * (n + 1) + i * n + j + HALF] = temp_d[i][j + HALF];
    d_dist[(Round * B + HALF) * (n + 1) + i * n + j] = temp_d[i + HALF][j + HALF];
}


__global__ void phase2_hor(int *d_dist, int Round, int n){
    int i = threadIdx.y;
    int j = threadIdx.x;
    if(blockIdx.x == Round) return;

    __shared__ int hor_d[B][B];
    // __shared__ int vert_d[B][B];
    __shared__ int pivot_d[B][B];
    int idx = blockIdx.x * B + j;

    //horizontal
    // if(blockIdx.x == 0){
    hor_d[i][j] = d_dist[(Round * B + i)*n + idx];
    hor_d[i + HALF][j] = d_dist[(Round * B + i + HALF)*n + idx];
    hor_d[i][j + HALF] = d_dist[(Round * B + i)*n + idx + HALF];
    hor_d[i+HALF][j+HALF] = d_dist[(Round * B + i)*n + HALF * (n + 1) + idx];

    pivot_d[i][j] = d_dist[(Round * B) * (n + 1) + i * n + j];
    pivot_d[i + HALF][j] = d_dist[(Round * B) * (n + 1) + (i + HALF) * n + j];
    pivot_d[i][j + HALF] = d_dist[(Round * B) * (n + 1) + i * n + j + HALF];
    pivot_d[i + HALF][j + HALF] = d_dist[(Round * B + HALF) * (n + 1) + i * n + j];

    __syncthreads();
    //shared memory
    #pragma unroll 32
    for(int k = 0; k < B; ++k){
        hor_d[i][j] = min(pivot_d[i][k] + hor_d[k][j], hor_d[i][j]);
        hor_d[i + HALF][j] = min(pivot_d[i + HALF][k] + hor_d[k][j], hor_d[i + HALF][j]);
        hor_d[i][j + HALF] = min(pivot_d[i][k] + hor_d[k][j + HALF], hor_d[i][j + HALF]);
        hor_d[i + HALF][j + HALF] = min(pivot_d[i + HALF][k] + hor_d[k][j + HALF], hor_d[i + HALF][j + HALF]);
    }

    d_dist[(Round * B + i)*n + idx] = hor_d[i][j];
    d_dist[(Round * B + i + HALF)*n + idx] = hor_d[i + HALF][j];
    d_dist[(Round * B + i)*n + idx + HALF] = hor_d[i][j + HALF] ;
    d_dist[(Round * B + i)*n + HALF * (n + 1) + idx] = hor_d[i+HALF][j+HALF];
}
__global__ void phase2(int *d_dist, int Round, int n){
    int i = threadIdx.y;
    int j = threadIdx.x;
    if(blockIdx.x == Round) return;

    __shared__ int hor_d[B][B];
    __shared__ int vert_d[B][B];
    __shared__ int pivot_d[B][B];
    int idx = blockIdx.x * B;
    // int flag = 0;

    //horizontal
    // if(blockIdx.y == 0){
        // flag = 1;
    hor_d[i][j] = d_dist[(Round * B + i)*n + idx + j];
    hor_d[i + HALF][j] = d_dist[(Round * B + i + HALF)*n + idx + j];
    hor_d[i][j + HALF] = d_dist[(Round * B + i)*n + idx + j + HALF];
    hor_d[i+HALF][j+HALF] = d_dist[(Round * B + i)*n + HALF * (n + 1) + idx + j];
    // } else {
    vert_d[i][j] = d_dist[(idx + i)*n + Round * B + j];
    vert_d[i + HALF][j] = d_dist[(idx + i + HALF)*n + Round * B + j];
    vert_d[i][j + HALF] = d_dist[(idx + i)*n + Round * B + j + HALF];
    vert_d[i+HALF][j+HALF] = d_dist[(idx + i)*n + Round * B + j + HALF * (n+1)];
    // }

    pivot_d[i][j] = d_dist[(Round * B) * (n + 1) + i * n + j];
    pivot_d[i + HALF][j] = d_dist[(Round * B) * (n + 1) + (i + HALF) * n + j];
    pivot_d[i][j + HALF] = d_dist[(Round * B) * (n + 1) + i * n + j + HALF];
    pivot_d[i + HALF][j + HALF] = d_dist[(Round * B + HALF) * (n + 1) + i * n + j];

    __syncthreads();
    //shared memory
    #pragma unroll 32
    for(int k = 0; k < B; ++k){
        hor_d[i][j] = min(pivot_d[i][k] + hor_d[k][j], hor_d[i][j]);
        hor_d[i + HALF][j] = min(pivot_d[i + HALF][k] + hor_d[k][j], hor_d[i + HALF][j]);
        hor_d[i][j + HALF] = min(pivot_d[i][k] + hor_d[k][j + HALF], hor_d[i][j + HALF]);
        hor_d[i + HALF][j + HALF] = min(pivot_d[i + HALF][k] + hor_d[k][j + HALF], hor_d[i + HALF][j + HALF]);

        vert_d[i][j] = min(vert_d[i][k] + pivot_d[k][j], vert_d[i][j]);
        vert_d[i + HALF][j] = min(vert_d[i + HALF][k] + pivot_d[k][j], vert_d[i + HALF][j]);
        vert_d[i][j + HALF] = min(vert_d[i][k] + pivot_d[k][j + HALF], vert_d[i][j + HALF]);
        vert_d[i + HALF][j + HALF] = min(vert_d[i + HALF][k] + pivot_d[k][j + HALF], vert_d[i + HALF][j + HALF]);
    }

        d_dist[(Round * B + i)*n + idx + j] = hor_d[i][j];
        d_dist[(Round * B + i + HALF)*n + idx + j] = hor_d[i + HALF][j];
        d_dist[(Round * B + i)*n + idx + j + HALF] = hor_d[i][j + HALF];
        d_dist[(Round * B + i)*n + HALF * (n + 1) + idx + j] = hor_d[i+HALF][j+HALF];

        d_dist[(idx + i)*n + Round * B + j] = vert_d[i][j];;
        d_dist[(idx + i + HALF)*n + Round * B + j] = vert_d[i + HALF][j];
        d_dist[(idx + i)*n + Round * B + j + HALF] = vert_d[i][j + HALF];
        d_dist[(idx + i)*n + Round * B + j + HALF * (n+1)] = vert_d[i+HALF][j+HALF];
}

__global__ void phase2_ver(int *d_dist, int Round, int n){
    int i = threadIdx.y;
    int j = threadIdx.x;
    if(blockIdx.y == Round) return;

    // __shared__ int hor_d[B][B];
    __shared__ int vert_d[B][B];
    __shared__ int pivot_d[B][B];
    int idx = blockIdx.y * B + i;

    vert_d[i][j] = d_dist[idx * n + Round * B + j];
    vert_d[i + HALF][j] = d_dist[(idx + HALF) * n + Round * B + j];
    vert_d[i][j + HALF] = d_dist[idx * n + Round * B + j + HALF];
    vert_d[i + HALF][j + HALF] = d_dist[idx * n + HALF * (n + 1) + Round * B + j];
    // }

    pivot_d[i][j] = d_dist[(Round * B) * (n + 1) + i * n + j];
    pivot_d[i + HALF][j] = d_dist[(Round * B) * (n + 1) + (i + HALF) * n + j];
    pivot_d[i][j + HALF] = d_dist[(Round * B) * (n + 1) + i * n + j + HALF];
    pivot_d[i + HALF][j + HALF] = d_dist[(Round * B + HALF) * (n + 1) + i * n + j];

    __syncthreads();
    //shared memory
    #pragma unroll 32
    for(int k = 0; k < B; ++k){

        vert_d[i][j] = min(vert_d[i][k] + pivot_d[k][j], vert_d[i][j]);
        vert_d[i + HALF][j] = min(vert_d[i + HALF][k] + pivot_d[k][j], vert_d[i + HALF][j]);
        vert_d[i][j + HALF] = min(vert_d[i][k] + pivot_d[k][j + HALF], vert_d[i][j + HALF]);
        vert_d[i + HALF][j + HALF] = min(vert_d[i + HALF][k] + pivot_d[k][j + HALF], vert_d[i + HALF][j + HALF]);
        // __syncthreads();
    }

    d_dist[idx * n + Round * B + j] = vert_d[i][j];
    d_dist[(idx + HALF) * n + Round * B + j] = vert_d[i + HALF][j];
    d_dist[idx * n + Round * B + j + HALF] = vert_d[i][j + HALF];
    d_dist[idx * n + HALF * (n + 1) + Round * B + j] = vert_d[i + HALF][j + HALF];

}
__global__ void phase3(int* d_dist, int Round, int n, int start){
    if(blockIdx.x == Round || (blockIdx.y + start) == Round) return;
    int i = threadIdx.y;
    int j = threadIdx.x;

    __shared__ int hor_d[B][B];
    __shared__ int vert_d[B][B];
    __shared__ int temp[B][B];
    int idxI = (blockIdx.y + start) * B + i;
    int idxJ = blockIdx.x * B + j;

    hor_d[i][j] = d_dist[(Round * B + i)*n + idxJ];
    hor_d[i + HALF][j] = d_dist[(Round * B + i + HALF)*n + idxJ];
    hor_d[i][j + HALF] = d_dist[(Round * B + i)*n + idxJ + HALF];
    hor_d[i + HALF][j + HALF] = d_dist[(Round * B + i + HALF)*n + idxJ + HALF];

    vert_d[i][j] = d_dist[idxI * n + Round * B + j];
    vert_d[i + HALF][j] = d_dist[(idxI + HALF) * n + Round * B + j];
    vert_d[i][j + HALF] = d_dist[idxI * n + Round * B + j + HALF];
    vert_d[i + HALF][j + HALF] = d_dist[(idxI + HALF) * n + Round * B + j + HALF];

    temp[i][j] = d_dist[idxI * n + idxJ];
    temp[i + HALF][j] = d_dist[(idxI + HALF) * n + idxJ];
    temp[i][j + HALF] = d_dist[idxI * n + idxJ + HALF];
    temp[i + HALF][j + HALF] = d_dist[idxI * n + HALF * (n + 1) + idxJ];
    __syncthreads();

    #pragma unroll 32
    for(int k = 0; k < B; ++k){
        temp[i][j] = min(temp[i][j], vert_d[i][k] + hor_d[k][j]);
        temp[i + HALF][j] = min(temp[i + HALF][j], vert_d[i + HALF][k] + hor_d[k][j]);
        temp[i][j + HALF] = min(temp[i][j + HALF], vert_d[i][k] + hor_d[k][j + HALF]);
        temp[i + HALF][j + HALF] = min(temp[i + HALF][j + HALF], vert_d[i + HALF][k] + hor_d[k][j + HALF]);
    }

    // d_dist[idxI * n + idxJ] = temp[i][j] * (temp[i][j] <= INF) + INF * (temp[i][j] > INF);
    // d_dist[(idxI + HALF) * n + idxJ] = temp[i + HALF][j] * (temp[i + HALF][j] <= INF) + INF * (temp[i+HALF][j] > INF);
    // d_dist[idxI * n + idxJ + HALF] = temp[i][j + HALF] * (temp[i][j + HALF] <= INF) + INF * (temp[i][j + HALF] > INF);
    // d_dist[idxI * n + HALF * (n + 1) + idxJ] = temp[i + HALF][j + HALF] * (temp[i + HALF][j + HALF] <= INF) + INF * (temp[i + HALF][j + HALF] > INF);
    d_dist[idxI * n + idxJ] = temp[i][j];
    d_dist[(idxI + HALF) * n + idxJ] = temp[i + HALF][j];
    d_dist[idxI * n + idxJ + HALF] = temp[i][j + HALF];
    d_dist[idxI * n + HALF * (n + 1) + idxJ] = temp[i + HALF][j + HALF];
}