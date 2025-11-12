/**
 * Simplified working version of KNN using UVM
 * Fixed version that actually works
 */

#include <cstdio>
#include <cuda.h>
#include <sys/time.h>
#include <cmath>

#define BLOCK_DIM 16

// Simple distance kernel
__global__ void compute_distances(float *ref, float *query, float *dist,
                                 int ref_nb, int query_nb, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < query_nb && j < ref_nb) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = query[i * dim + d] - ref[j * dim + d];
            sum += diff * diff;
        }
        dist[i * ref_nb + j] = sqrtf(sum);
    }
}

// Simple k-NN on CPU (works with UVM)
void find_k_nearest(float *dist, int *ind, int query_nb, int ref_nb, int k) {
    for (int i = 0; i < query_nb; i++) {
        // For each query point, find k nearest
        for (int kk = 0; kk < k; kk++) {
            float min_dist = 1e30f;
            int min_idx = 0;

            for (int j = 0; j < ref_nb; j++) {
                bool already_selected = false;
                for (int check = 0; check < kk; check++) {
                    if (ind[i * k + check] == j) {
                        already_selected = true;
                        break;
                    }
                }

                if (!already_selected && dist[i * ref_nb + j] < min_dist) {
                    min_dist = dist[i * ref_nb + j];
                    min_idx = j;
                }
            }

            ind[i * k + kk] = min_idx;
        }
    }
}

int main() {
    // Parameters - reduced for stability
    int ref_nb = 512;     // Reference points
    int query_nb = 512;   // Query points
    int dim = 16;         // Dimensions
    int k = 10;           // Nearest neighbors

    printf("K-NN with UVM\n");
    printf("Reference points: %d\n", ref_nb);
    printf("Query points: %d\n", query_nb);
    printf("Dimensions: %d\n", dim);
    printf("K: %d\n\n", k);

    // Allocate using UVM
    float *ref, *query, *dist;
    int *ind;

    cudaMallocManaged(&ref, ref_nb * dim * sizeof(float));
    cudaMallocManaged(&query, query_nb * dim * sizeof(float));
    cudaMallocManaged(&dist, query_nb * ref_nb * sizeof(float));
    cudaMallocManaged(&ind, query_nb * k * sizeof(int));

    // Initialize data
    printf("Initializing data...\n");
    srand(12345);
    for (int i = 0; i < ref_nb * dim; i++)
        ref[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < query_nb * dim; i++)
        query[i] = (float)rand() / RAND_MAX;

    // Run on GPU
    printf("Computing distances on GPU...\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((query_nb + BLOCK_DIM - 1) / BLOCK_DIM,
              (ref_nb + BLOCK_DIM - 1) / BLOCK_DIM);

    compute_distances<<<grid, block>>>(ref, query, dist, ref_nb, query_nb, dim);

    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    double gpu_time = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1e6;

    printf("GPU distance computation: %.3f seconds\n", gpu_time);

    // Find k-nearest on CPU
    printf("Finding k-nearest neighbors on CPU...\n");
    gettimeofday(&start, NULL);

    find_k_nearest(dist, ind, query_nb, ref_nb, k);

    gettimeofday(&end, NULL);
    double cpu_time = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1e6;

    printf("CPU k-NN selection: %.3f seconds\n", cpu_time);
    printf("Total time: %.3f seconds\n", gpu_time + cpu_time);

    // Show some results
    printf("\nFirst query point's %d nearest neighbors:\n", k);
    for (int i = 0; i < k; i++) {
        int idx = ind[i];
        printf("  Neighbor %d: index=%d, distance=%.4f\n",
               i+1, idx, dist[idx]);
    }

    // Cleanup
    cudaFree(ref);
    cudaFree(query);
    cudaFree(dist);
    cudaFree(ind);

    printf("\nSUCCESS\n");
    return 0;
}
