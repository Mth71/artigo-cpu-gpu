#!/usr/bin/env bash
# Roda todos os experimentos e grava:
#   ../artigo/resultados/dados.csv     (uma linha por medição)
#   ../artigo/resultados/ambiente.txt  (hardware e software, para a Tabela 1)
#
# Variáveis opcionais:
#   REPS=30 AQ=5 TAMANHOS="64 128 ..." SEQ_MAX=1024 THREADS=<núcleos físicos>
#
# Antes de rodar: feche navegador/IDE, deixe o notebook na tomada e, se puder,
#   sudo cpupower frequency-set -g performance
set -euo pipefail
cd "$(dirname "$0")"

REPS=${REPS:-30}
AQ=${AQ:-5}
TAMANHOS=${TAMANHOS:-"64 128 256 512 1024 2048 4096"}
SEQ_MAX=${SEQ_MAX:-1024}   # o ijk ingênuo fica lento demais acima disso
SEQ_REPS=${SEQ_REPS:-10}
FISICOS=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
THREADS=${THREADS:-$FISICOS}

OUT=../artigo/resultados
mkdir -p "$OUT"
CSV="$OUT/dados.csv"
LOG="$OUT/verificacao.log"

export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close OMP_PLACES=cores
export NVIDIA_TF32_OVERRIDE=0

# ---- ambiente -------------------------------------------------------------
{
  echo "data: $(date -Iseconds)"
  echo "threads CPU usadas: $THREADS (núcleos físicos: $FISICOS)"
  echo; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L2|L3'
  echo; free -h | head -2
  echo; uname -a
  echo; gcc --version | head -1
  echo; (nvcc --version | tail -2) || true
  echo; (nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max --format=csv) || true
  echo; (cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) 2>/dev/null || true
  echo; dpkg -l | grep -E 'libopenblas|cuda-toolkit|libcublas' | awk '{print $2, $3}' || true
  echo; echo "REPS=$REPS AQ=$AQ TAMANHOS=$TAMANHOS"
} > "$OUT/ambiente.txt"

echo "impl,n,rep,metrica,tempo_s" > "$CSV"
: > "$LOG"

rodar() {
  echo ">> $*" >&2
  "$@" >> "$CSV" 2>> "$LOG" || echo "   (saiu com código $?)" | tee -a "$LOG" >&2
  tail -1 "$LOG" >&2
}

for n in $TAMANHOS; do
  if [ "$n" -le "$SEQ_MAX" ]; then
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 rodar ./mm_cpu seq "$n" "$SEQ_REPS" 1
  fi
  rodar ./mm_cpu omp  "$n" "$REPS" "$AQ"
  OMP_PROC_BIND=false rodar ./mm_cpu blas "$n" "$REPS" "$AQ"
  rodar ./mm_gpu naive  "$n" "$REPS" "$AQ"
  rodar ./mm_gpu tiled  "$n" "$REPS" "$AQ"
  rodar ./mm_gpu cublas "$n" "$REPS" "$AQ"
  rodar ./mm_gpu cublas "$n" "$REPS" "$AQ" pinned
done

if grep -q FALHOU "$LOG"; then
  echo "ATENÇÃO: alguma verificação de corretude falhou, veja $LOG" >&2
  exit 2
fi
echo "Pronto: $CSV  — agora rode: python3 analisar.py" >&2
