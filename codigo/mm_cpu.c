/*
 * mm_cpu.c - multiplicação de matrizes C = A x B (FP32, row-major) na CPU.
 *
 * Implementações:
 *   seq   - laço ijk ingênuo, 1 thread (linha de base "ingênua", como nos
 *           trabalhos que reportam speedups de 100x)
 *   omp   - blocagem em cache + ordem ikj vetorizável + OpenMP (multithread)
 *   blas  - cblas_sgemm da OpenBLAS (multithread, estado da arte em CPU)
 *           [só se compilado com -DUSE_OPENBLAS]
 *
 * Uso: ./mm_cpu <impl> <n> <repeticoes> <aquecimento>
 * Saída: CSV  impl,n,rep,metrica,tempo_s   (metrica = "total")
 */
#include "comum.h"
#include <omp.h>
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifndef BS
#define BS 64 /* tamanho do bloco; 3 blocos de 64x64 floats = 48 KiB */
#endif

static void mm_seq(const float *A, const float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++)
                s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

static void mm_omp(const float *restrict A, const float *restrict B,
                   float *restrict C, int n)
{
    /* Cada thread fica com um conjunto disjunto de faixas de linhas de C:
     * não há escrita compartilhada, logo não há corrida. */
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += BS) {
        int iMax = ii + BS < n ? ii + BS : n;
        for (int i = ii; i < iMax; i++)
            memset(&C[(size_t)i * n], 0, sizeof(float) * (size_t)n);
        for (int kk = 0; kk < n; kk += BS) {
            int kMax = kk + BS < n ? kk + BS : n;
            for (int jj = 0; jj < n; jj += BS) {
                int jMax = jj + BS < n ? jj + BS : n;
                for (int i = ii; i < iMax; i++) {
                    float *restrict c = &C[(size_t)i * n];
                    for (int k = kk; k < kMax; k++) {
                        const float a = A[(size_t)i * n + k];
                        const float *restrict b = &B[(size_t)k * n];
#pragma omp simd
                        for (int j = jj; j < jMax; j++)
                            c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

#ifdef USE_OPENBLAS
static void mm_blas(const float *A, const float *B, float *C, int n)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                1.0f, A, n, B, n, 0.0f, C, n);
}
#endif

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "uso: %s <seq|omp|blas> <n> <reps> <aquecimento>\n", argv[0]);
        return 1;
    }
    const char *impl = argv[1];
    int n = atoi(argv[2]), reps = atoi(argv[3]), aq = atoi(argv[4]);

    void (*f)(const float *, const float *, float *, int) = NULL;
    if (!strcmp(impl, "seq")) f = mm_seq;
    else if (!strcmp(impl, "omp")) f = mm_omp;
#ifdef USE_OPENBLAS
    else if (!strcmp(impl, "blas")) f = mm_blas;
#endif
    if (!f) { fprintf(stderr, "implementacao indisponivel: %s\n", impl); return 1; }

    size_t len = (size_t)n * n;
    float *A = aligned_alloc(64, ((len * sizeof(float) + 63) / 64) * 64);
    float *B = aligned_alloc(64, ((len * sizeof(float) + 63) / 64) * 64);
    float *C = aligned_alloc(64, ((len * sizeof(float) + 63) / 64) * 64);
    if (!A || !B || !C) { fprintf(stderr, "sem memoria\n"); return 1; }
    preencher(A, len, 1u);
    preencher(B, len, 2u);

    for (int r = 0; r < aq; r++) f(A, B, C, n); /* aquecimento: descartado */

    for (int r = 0; r < reps; r++) {
        memset(C, 0, len * sizeof(float));
        double t0 = agora_s();
        f(A, B, C, n);
        double t1 = agora_s();
        printf("cpu_%s,%d,%d,total,%.9f\n", impl, n, r, t1 - t0);
        fflush(stdout);
    }

    double err = verificar(A, B, C, n);
    fprintf(stderr, "[cpu_%s n=%d threads=%d] erro_rel_max=%.2e %s\n", impl, n,
            omp_get_max_threads(), err, err < TOLERANCIA ? "OK" : "FALHOU");
    free(A); free(B); free(C);
    return err < TOLERANCIA ? 0 : 2;
}
