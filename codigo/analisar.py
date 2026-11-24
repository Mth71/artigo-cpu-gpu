#!/usr/bin/env python3
"""
Lê ../artigo/resultados/dados.csv e gera, prontos para o LaTeX:
  ../artigo/figuras/tempo.pdf        tempo x N (log-log)
  ../artigo/figuras/speedup.pdf      speedup da GPU sobre a melhor CPU x N
  ../artigo/figuras/decomposicao.pdf fração do tempo ponta a ponta (cuBLAS)
  ../artigo/resultados/tabela.tex    tabela com média ± IC95%
  ../artigo/resultados/macros.tex    números citados no texto (\\breakeven etc.)
  ../artigo/resultados/resumo.csv    estatísticas completas

Dependências: python3-numpy python3-pandas python3-scipy python3-matplotlib
"""
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

AQUI = pathlib.Path(__file__).resolve().parent
RES = AQUI.parent / "artigo" / "resultados"
FIG = AQUI.parent / "artigo" / "figuras"
CSV = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "dados.csv"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# Cores fixas por entidade (paleta categórica validada p/ daltonismo) +
# marcador e traço como codificação secundária (impressão em P&B).
ESTILO = {
    "cpu_seq":           dict(cor="#008300", m="v", rot="CPU ingênua (1 thread)"),
    "cpu_omp":           dict(cor="#2a78d6", m="o", rot="CPU OpenMP blocada"),
    "cpu_blas":          dict(cor="#eb6834", m="s", rot="CPU OpenBLAS"),
    "gpu_naive":         dict(cor="#e87ba4", m="^", rot="GPU ingênua"),
    "gpu_tiled":         dict(cor="#eda100", m="D", rot="GPU mem. compartilhada"),
    "gpu_cublas":        dict(cor="#1baf7a", m="P", rot="GPU cuBLAS"),
    "gpu_cublas_pinned": dict(cor="#1baf7a", m="X", rot="GPU cuBLAS (pinned)"),
}
TINTA, TINTA2, GRADE = "#1f1f1e", "#5f5e58", "#e4e3de"
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
LARG = 3.4  # polegadas ~ largura útil de meia página no template SBC


def eixo_n(ax, ns):
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def ic95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    rng = np.random.default_rng(7)
    b = np.median(rng.choice(x, (2000, len(x))), axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (hi - lo) / 2


def bootstrap_razao(num, den, b=4000, seed=7):
    """IC95% (percentil) da razão de médias mean(num)/mean(den)."""
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num), np.asarray(den)
    r = [np.median(rng.choice(num, len(num))) / np.median(rng.choice(den, len(den)))
         for _ in range(b)]
    return np.percentile(r, [2.5, 97.5])


def fmt_t(s):
    if s < 1e-3:
        return f"{s*1e6:.1f}\\,$\\mu$s"
    if s < 1:
        return f"{s*1e3:.2f}\\,ms"
    return f"{s:.2f}\\,s"


def main():
    df = pd.read_csv(CSV)
    g = df.groupby(["impl", "n", "metrica"])["tempo_s"]
    res = g.agg(media="median", media_arit="mean", dp="std", mediana="median", reps="count").reset_index()
    res["ic95"] = g.apply(ic95).values
    res["cv"] = res["dp"] / res["media_arit"]
    res["gflops"] = 2 * res["n"].astype(float) ** 3 / res["media"] / 1e9
    res.to_csv(RES / "resumo.csv", index=False)

    def serie(impl, met):
        s = res[(res.impl == impl) & (res.metrica == met)].sort_values("n")
        return s

    def amostras(impl, met, n):
        return df[(df.impl == impl) & (df.metrica == met) & (df.n == n)]["tempo_s"]

    impls = set(res.impl)
    cpu_ref = "cpu_blas" if "cpu_blas" in impls else "cpu_omp"
    gpu_ref = "gpu_cublas" if "gpu_cublas" in impls else "gpu_tiled"
    Ns = sorted(set(serie(cpu_ref, "total").n) & set(serie(gpu_ref, "e2e").n))

    # ---------------- Figura 1: tempo x N ----------------
    fig, ax = plt.subplots(figsize=(LARG, 2.4))
    curvas = [("cpu_seq", "total", "-"), ("cpu_omp", "total", "-"),
              ("cpu_blas", "total", "-"), ("gpu_naive", "e2e", "-"),
              ("gpu_tiled", "e2e", "-"), ("gpu_cublas", "e2e", "-"),
              ("gpu_cublas", "kernel", "--")]
    for impl, met, ls in curvas:
        s = serie(impl, met)
        if s.empty:
            continue
        e = ESTILO[impl]
        rot = e["rot"] + (" (só kernel)" if met == "kernel" else "")
        ax.errorbar(s.n, s.media, yerr=s.ic95, color=e["cor"], marker=e["m"],
                    ls=ls, label=rot, capsize=0, elinewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ordem da matriz $N$")
    ax.set_ylabel("Tempo (s, escala log)")
    eixo_n(ax, sorted(set(res.n)))
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.savefig(FIG / "tempo.pdf")
    plt.close(fig)

    # ---------------- Figura 2: speedup ----------------
    linhas_sp = {}
    fig, ax = plt.subplots(figsize=(LARG, 2.3))
    for met, ls, rot in [("e2e", "-", "ponta a ponta"), ("kernel", "--", "só kernel")]:
        sp, lo, hi = [], [], []
        for n in Ns:
            c, gg = amostras(cpu_ref, "total", n), amostras(gpu_ref, met, n)
            sp.append(c.median() / gg.median())
            a, b = bootstrap_razao(c, gg)
            lo.append(sp[-1] - a); hi.append(b - sp[-1])
        linhas_sp[met] = sp
        ax.errorbar(Ns, sp, yerr=[lo, hi], color=ESTILO[gpu_ref]["cor"],
                    marker=ESTILO[gpu_ref]["m"], ls=ls, label=rot, elinewidth=0.8)
    ax.axhline(1.0, color=TINTA2, lw=0.9)
    ax.text(Ns[-1], 1.0, "empate (speedup = 1) ", va="top", ha="right",
            fontsize=6, color=TINTA2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Ordem da matriz $N$")
    ax.set_ylabel("Speedup (escala log)")
    eixo_n(ax, Ns)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2,
              title=f"{ESTILO[gpu_ref]['rot']} vs. {ESTILO[cpu_ref]['rot']}",
              title_fontsize=6)
    fig.savefig(FIG / "speedup.pdf")
    plt.close(fig)

    # ---------------- Figura 3: decomposição ----------------
    fig, ax = plt.subplots(figsize=(LARG, 2.1))
    partes = [("h2d", "Cópia CPU→GPU", "#2a78d6"), ("kernel", "Kernel", "#eb6834"),
              ("d2h", "Cópia GPU→CPU", "#1baf7a")]
    base = np.zeros(len(Ns))
    tot = np.array([sum(amostras(gpu_ref, p, n).median() for p, _, _ in partes) for n in Ns])
    frac = {}
    x = np.arange(len(Ns))
    for p, rot, cor in partes:
        v = np.array([amostras(gpu_ref, p, n).median() for n in Ns]) / tot * 100
        frac[p] = v
        ax.bar(x, v, bottom=base, color=cor, label=rot, width=0.7,
               edgecolor="white", linewidth=1.0)
        base += v
    ax.set_xticks(x, [str(n) for n in Ns])
    ax.set_xlabel("Ordem da matriz $N$")
    ax.set_ylabel("% do tempo")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(FIG / "decomposicao.pdf")
    plt.close(fig)

    # ---------------- Tabela ----------------
    cols = [("cpu_seq", "total"), ("cpu_omp", "total"), ("cpu_blas", "total"),
            ("gpu_tiled", "e2e"), ("gpu_cublas", "kernel"), ("gpu_cublas", "e2e")]
    cols = [c for c in cols if c[0] in impls]
    cab = {"cpu_seq": "CPU 1T", "cpu_omp": "CPU OMP", "cpu_blas": "OpenBLAS",
           "gpu_tiled": "GPU tiled", "gpu_cublas": "cuBLAS"}
    linhas = []
    linhas.append("\\begin{tabular}{r" + "r" * len(cols) + "}")
    linhas.append("\\hline")
    linhas.append("$N$ & " + " & ".join(
        cab[i] + (" (kernel)" if m == "kernel" else " (p.~a~p.)" if m == "e2e" else "")
        for i, m in cols) + " \\\\ \\hline")
    for n in sorted(set(res.n)):
        cel = []
        for i, m in cols:
            s = res[(res.impl == i) & (res.metrica == m) & (res.n == n)]
            if s.empty:
                cel.append("--")
            else:
                r = s.iloc[0]
                cel.append(f"{fmt_t(r.media)} $\\pm${100*r.ic95/r.media:.1f}\\%")
        linhas.append(f"{n} & " + " & ".join(cel) + " \\\\")
    linhas.append("\\hline\n\\end{tabular}")
    (RES / "tabela.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    # ---------------- Macros para o texto ----------------
    e2e = linhas_sp["e2e"]
    ganha = [n for n, s in zip(Ns, e2e) if s > 1]
    if not ganha:
        be = f"não foi atingido até $N={Ns[-1]}$"
        be_en = f"was not reached up to $N={Ns[-1]}$"
    else:
        # menor N a partir do qual a GPU vence em todos os tamanhos maiores
        k = next(i for i in range(len(Ns)) if all(s > 1 for s in e2e[i:]))
        be = f"$N={Ns[k]}$" if k == 0 else f"entre $N={Ns[k-1]}$ e $N={Ns[k]}$"
        be_en = f"$N={Ns[k]}$" if k == 0 else f"between $N={Ns[k-1]}$ and $N={Ns[k]}$"
    m = {
        "breakeven": be,
        "breakevenEn": be_en,
        "speedupEteMax": f"{max(e2e):.1f}",
        "speedupEteMin": f"{min(e2e):.2f}",
        "speedupKernelMax": f"{max(linhas_sp['kernel']):.1f}",
        "nMax": str(Ns[-1]),
        "nMin": str(Ns[0]),
        "fracTransfMin": f"{(frac['h2d'][0] + frac['d2h'][0]):.0f}",
        "fracTransfMax": f"{(frac['h2d'][-1] + frac['d2h'][-1]):.0f}",
        "cvMax": f"{100*res[res.reps >= 10].cv.max():.1f}",
        "gflopsCpu": f"{serie(cpu_ref, 'total').gflops.max():.0f}",
        "gflopsGpu": f"{serie(gpu_ref, 'kernel').gflops.max():.0f}",
    }
    # Modelo analítico simples (Seção 3 do artigo): a GPU vence quando
    #   2N^3/Pcpu > 12N^2/B + 2N^3/Pgpu  =>  N* = 6 Pcpu / (B (1 - Pcpu/Pgpu))
    nn = Ns[-1]
    p_cpu = serie(cpu_ref, "total").gflops.max() * 1e9
    p_gpu = serie(gpu_ref, "kernel").gflops.max() * 1e9
    b_pcie = (12.0 * nn * nn) / (amostras(gpu_ref, "h2d", nn).median()
                                 + amostras(gpu_ref, "d2h", nn).median())
    m["banda"] = f"{b_pcie/1e9:.1f}"
    m["nEstimado"] = (f"{6*p_cpu/(b_pcie*(1-p_cpu/p_gpu)):.0f}" if p_gpu > p_cpu
                      else "indefinido")
    seq = serie("cpu_seq", "total")
    if not seq.empty:
        nseq = int(seq.n.max())
        t_seq = amostras("cpu_seq", "total", nseq).median()
        m["nSeq"] = str(nseq)
        m["speedupSeqKernel"] = f"{t_seq / amostras(gpu_ref, 'kernel', nseq).median():.0f}"
        m["speedupSeqEte"] = f"{t_seq / amostras(gpu_ref, 'e2e', nseq).median():.0f}"
        m["speedupRefEteSeqN"] = f"{amostras(cpu_ref, 'total', nseq).median() / amostras(gpu_ref, 'e2e', nseq).median():.1f}"
    if "gpu_cublas_pinned" in impls:
        nn = Ns[-1]
        m["ganhoPinned"] = f"{100*(1 - amostras('gpu_cublas_pinned','e2e',nn).median()/amostras('gpu_cublas','e2e',nn).median()):.0f}"
    out = ["% Gerado por codigo/analisar.py -- nao edite a mao."]
    out += [f"\\renewcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (RES / "macros.tex").write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"CPU de referência: {cpu_ref} | GPU de referência: {gpu_ref}")
    for k, v in m.items():
        print(f"  {k:20s} {v}")
    print(f"Figuras em {FIG}, tabela e macros em {RES}")


if __name__ == "__main__":
    main()
