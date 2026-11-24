/*
 * mm_gpu.cu - multiplicação de matrizes C = A x B (FP32, row-major) na GPU.
 *
 * Implementações:
 *   naive   - 1 thread por elemento de C, leituras direto da memória global
 *   tiled   - blocagem em memória compartilhada (TILE x TILE), cf. Hwu et al.
 *   cublas  - cublasSgemm (biblioteca do fabricante; TF32 desligado)
 *
 * Em cada repetição medimos, com cudaEvent:
 *   h2d    - cópia de A e B da CPU para a GPU
 *   kernel - só a computação
 *   d2h    - cópia de C de volta para a CPU
 * e, com relógio de parede (CLOCK_MONOTONIC):
 *   e2e    - ponta a ponta: h2d + kernel + d2h + sincronização
 * A alocação na GPU e a criação do contexto/handle ficam FORA da medição
 * (custo único, amortizado numa aplicação real) e isso é dito no artigo.
 *
 * Uso: ./mm_gpu <impl> <n> <repeticoes> <aquecimento> [pageable|pinned]
 * Saída: CSV  impl,n,rep,metrica,tempo_s
 */
#include "comum.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(3); } } while (0)
#define CKB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
    exit(3); } } while (0)

#ifndef TILE
#define TILE 32
#endif

__global__ void k_naive(const float *__restrict__ A, const float *__restrict__ B,
                        float *__restrict__ C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.0f;
        for (int k = 0; k < n; k++)
            s += A[(size_t)row * n + k] * B[(size_t)k * n + col];
        C[(size_t)row * n + col] = s;
    }
}

__global__ void k_tiled(const float *__restrict__ A, const float *__restrict__ B,
                        float *__restrict__ C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float s = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int ka = t * TILE + tx, kb = t * TILE + ty;
        As[ty][tx] = (row < n && ka < n) ? A[(size_t)row * n + ka] : 0.0f;
        Bs[ty][tx] = (kb < n && col < n) ? B[(size_t)kb * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++)
            s += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < n && col < n)
        C[(size_t)row * n + col] = s;
}

enum Impl { NAIVE, TILED, CUBLAS };

static void computar(Impl impl, cublasHandle_t h, const float *dA,
                     const float *dB, float *dC, int n)
{
    if (impl == CUBLAS) {
        /* cuBLAS é column-major: C^T = B^T A^T  =>  passar B antes de A. */
        const float alfa = 1.0f, beta = 0.0f;
        CKB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                        &alfa, dB, n, dA, n, &beta, dC, n));
    } else {
        dim3 bloco(TILE, TILE);
        dim3 grade((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        if (impl == NAIVE) k_naive<<<grade, bloco>>>(dA, dB, dC, n);
        else               k_tiled<<<grade, bloco>>>(dA, dB, dC, n);
        CK(cudaGetLastError());
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "uso: %s <naive|tiled|cublas> <n> <reps> <aquecimento> [pageable|pinned]\n", argv[0]);
        return 1;
    }
    const char *nome = argv[1];
    int n = atoi(argv[2]), reps = atoi(argv[3]), aq = atoi(argv[4]);
    int pinned = (argc > 5 && !strcmp(argv[5], "pinned"));

    Impl impl;
    if (!strcmp(nome, "naive")) impl = NAIVE;
    else if (!strcmp(nome, "tiled")) impl = TILED;
    else if (!strcmp(nome, "cublas")) impl = CUBLAS;
    else { fprintf(stderr, "implementacao desconhecida: %s\n", nome); return 1; }

    size_t len = (size_t)n * n, bytes = len * sizeof(float);
    float *A, *B, *C;
    if (pinned) {
        CK(cudaMallocHost(&A, bytes)); CK(cudaMallocHost(&B, bytes)); CK(cudaMallocHost(&C, bytes));
    } else {
        A = (float *)malloc(bytes); B = (float *)malloc(bytes); C = (float *)malloc(bytes);
        if (!A || !B || !C) { fprintf(stderr, "sem memoria\n"); return 1; }
    }
    preencher(A, len, 1u);
    preencher(B, len, 2u);

    /* Custos únicos, fora da medição: contexto, alocação e handle. */
    CK(cudaFree(0));
    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    cublasHandle_t h = NULL;
    if (impl == CUBLAS) {
        CKB(cublasCreate(&h));
        CKB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)); /* sem TF32 */
    }
    cudaEvent_t e0, e1, e2, e3;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));

    const char *sufixo = pinned ? "_pinned" : "";
    for (int r = -aq; r < reps; r++) {        /* r < 0: aquecimento */
        CK(cudaMemset(dC, 0, bytes));
        CK(cudaDeviceSynchronize());
        double t0 = agora_s();
        CK(cudaEventRecord(e0));
        CK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e1));
        computar(impl, h, dA, dB, dC, n);
        CK(cudaEventRecord(e2));
        CK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(e3));
        CK(cudaEventSynchronize(e3));
        double t1 = agora_s();
        if (r < 0) continue;

        float ms_h2d, ms_k, ms_d2h;
        CK(cudaEventElapsedTime(&ms_h2d, e0, e1));
        CK(cudaEventElapsedTime(&ms_k, e1, e2));
        CK(cudaEventElapsedTime(&ms_d2h, e2, e3));
        printf("gpu_%s%s,%d,%d,h2d,%.9f\n", nome, sufixo, n, r, ms_h2d * 1e-3);
        printf("gpu_%s%s,%d,%d,kernel,%.9f\n", nome, sufixo, n, r, ms_k * 1e-3);
        printf("gpu_%s%s,%d,%d,d2h,%.9f\n", nome, sufixo, n, r, ms_d2h * 1e-3);
        printf("gpu_%s%s,%d,%d,e2e,%.9f\n", nome, sufixo, n, r, t1 - t0);
        fflush(stdout);
    }

    double err = verificar(A, B, C, n);
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "[gpu_%s%s n=%d %s] erro_rel_max=%.2e %s\n", nome, sufixo, n,
            p.name, err, err < TOLERANCIA ? "OK" : "FALHOU");

    if (h) cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (pinned) { cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C); }
    else { free(A); free(B); free(C); }
    return err < TOLERANCIA ? 0 : 2;
}
