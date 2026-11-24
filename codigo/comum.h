/*
 * comum.h - utilitários compartilhados entre os benchmarks de CPU e GPU.
 *
 * Saída padronizada em CSV (stdout), uma linha por medição:
 *   impl,n,rep,metrica,tempo_s
 * metrica: "total" (CPU), "kernel", "h2d", "d2h", "e2e" (GPU)
 */
#ifndef COMUM_H
#define COMUM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static inline double agora_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Gerador determinístico (xorshift32) para que CPU e GPU usem as mesmas
 * matrizes, independentemente da libc. Valores uniformes em [-1, 1]. */
static inline void preencher(float *m, size_t len, unsigned semente)
{
    unsigned x = semente ? semente : 2463534242u;
    for (size_t i = 0; i < len; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        m[i] = (float)((double)x / 4294967295.0 * 2.0 - 1.0);
    }
}

/* Verificação por amostragem: recalcula AMOSTRAS elementos de C em double e
 * compara o erro absoluto com sum_k |a_ik * b_kj| (escala natural do erro de
 * arredondamento). Retorna o maior erro relativo encontrado. */
#define AMOSTRAS 64
static inline double verificar(const float *A, const float *B, const float *C, int n)
{
    double pior = 0.0;
    unsigned x = 12345u;
    for (int s = 0; s < AMOSTRAS; s++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        int i = (int)(x % (unsigned)n);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        int j = (int)(x % (unsigned)n);
        double ref = 0.0, escala = 0.0;
        for (int k = 0; k < n; k++) {
            double p = (double)A[(size_t)i * n + k] * (double)B[(size_t)k * n + j];
            ref += p;
            escala += fabs(p);
        }
        double err = fabs((double)C[(size_t)i * n + j] - ref) / (escala > 0 ? escala : 1.0);
        if (err > pior) pior = err;
    }
    return pior;
}
/* Limite folgado para FP32: o erro típico cresce ~ sqrt(n)*eps; um resultado
 * errado (índice trocado, corrida, etc.) dá erro relativo da ordem de 1. */
#define TOLERANCIA 1e-3

#endif
