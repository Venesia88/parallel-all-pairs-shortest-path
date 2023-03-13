#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <algorithm>
using namespace std;

const int INF = ((1 << 30) - 1);
// const int V = 50010;
// void input(char* inFileName);
// void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
// static int Dist[V][V];
int **Dist;
// vector <vector <int>> Dist;
__m128i zero = _mm_setzero_si128();

union simd {
    alignas(16) int vs[4];
    __m128i v;
};

int main(int argc, char* argv[]) {
    // input(argv[1]);
    FILE* file = fopen(argv[1], "rb"); 
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    Dist = (int**)malloc(n * sizeof(int*));

    // for(int i = 0; i<n; ++i){
    //     // for(int j = 0; j < n; ++j)
    //     //     printf("%d\n", Dist[i][j]);
    //     Dist[i] = (int*)malloc(n * sizeof(int));
    //     for(int j = 0; j < n; ++j){
    //         if (i == j) {
    //             Dist[i][j] = 0;
    //         } else {
    //             Dist[i][j] = INF;
    //         }
    //     }
    // }

    // for(int i = 0; i<n; ++i) Dist[i] = (int*)malloc(n * sizeof(int));

    #pragma unroll 64
    for(int i = 0; i<n; ++i){
        // #pragma unroll(4)
        Dist[i] = (int*)malloc(n * sizeof(int));
        for(int j = 0; j < n; ++j){
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }
    



    // for(int i = 0; i < n; ++i){
    //     // printf("%d\n", row[i]);
    //     Dist[i][i] = 0;
    // }
    int *buf = (int*)malloc(m*3*sizeof(int));

    fread(buf, sizeof(int), m * 3, file);
    fclose(file);

    // int pair[3];
    // #pragma omp parallel for
    #pragma unroll 32
    for (int i = 0; i < m; ++i) {
        Dist[buf[i * 3]][buf[i * 3 + 1]] = buf[i*3 + 2];
    }
    
    
    int B = 512;
    block_FW(B);
    // output(argv[2]);
    FILE* outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        // for (int j = 0; j < n; ++j) {
        //     if (Dist[i][j] >= INF) Dist[i][j] = INF;
        // }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
    return 0;
}

void block_FW(int B) {
    int round = (n + B - 1) / B;
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        //left
        cal(B, r, r, 0, r, 1);
        //down
        cal(B, r, r, r + 1, round - r - 1, 1);
        //up
        cal(B, r, 0, r, 1, r);
        //right
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times

            int block_internal_start_x = b_i * B;
            int block_internal_end_x = (b_i + 1) * B;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;

            if (block_internal_end_x > n) block_internal_end_x = n;
            if (block_internal_end_y > n) block_internal_end_y = n;
            
            
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {

                #pragma omp parallel 
                {
                    simd d, b, cmp;
                    __m128i a;
                    #pragma omp for collapse(2) schedule(static)
                    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                        for (int j = block_internal_start_y; j < block_internal_end_y; j+=4) {
                            a = _mm_set1_epi32(Dist[i][k]);
                            b.v = _mm_loadu_si128((__m128i const*) &Dist[k][j]);
                            d.v = _mm_loadu_si128((__m128i const*) &Dist[i][j]);
                            b.v = _mm_add_epi32(a, b.v);
                            cmp.v = _mm_cmplt_epi32(b.v, d.v);
                            if(cmp.vs[0]) Dist[i][j] = min(b.vs[0], INF);
                            if(cmp.vs[1]) Dist[i][j + 1] = min(b.vs[1], INF);
                            if(cmp.vs[2]) Dist[i][j + 2] = min(b.vs[2], INF);
                            if(cmp.vs[3]) Dist[i][j + 3] = min(b.vs[3], INF);
                        }
                    }
                }
            }
        }
    }
}
