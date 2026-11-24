# Quando a GPU compensa? — artigo + experimentos

```
artigo/
  main.tex            texto completo no formato do template SBC
  referencias.bib     as 5 referências do grupo + 4 de apoio
  figuras/            (vazia) gerada por codigo/analisar.py
  resultados/         (vazia) dados.csv, ambiente.txt, tabela.tex, macros.tex
codigo/
  comum.h             geração das matrizes, verificação de corretude
  mm_cpu.c            CPU: ingênua 1 thread, OpenMP blocada, OpenBLAS
  mm_gpu.cu           GPU: ingênua, tiled (mem. compartilhada), cuBLAS
  Makefile
  rodar_experimentos.sh
  analisar.py         estatística (média, IC 95%), figuras, tabela e números do texto
```

Os números do texto (ponto de equilíbrio, speedups, % de transferência...) são
macros. Enquanto os experimentos não rodam, aparecem como **[a medir]** em
vermelho. Os trechos em **laranja** são interpretações que precisam ser
conferidas com os dados reais. Antes de submeter, troque `\revisartrue` por
`\revisarfalse` no `main.tex`.

## 1. Preparar o Ubuntu

```bash
sudo apt update
sudo apt install build-essential libopenblas-dev python3-numpy python3-pandas \
                 python3-scipy python3-matplotlib
# Driver NVIDIA + CUDA Toolkit (o mais simples no Ubuntu):
sudo ubuntu-drivers install
sudo apt install nvidia-cuda-toolkit      # ou o toolkit oficial da NVIDIA
# reinicie e confira:
nvidia-smi
nvcc --version
```

## 2. Compilar e testar rápido

```bash
cd codigo
make                       # se -arch=native falhar: make SM=86   (RTX 30xx; 89 = RTX 40xx; 75 = GTX 16xx/RTX 20xx)
./mm_cpu blas 512 3 1
./mm_gpu cublas 512 3 1
```
Cada execução imprime no fim `erro_rel_max=... OK`. Se aparecer `FALHOU`, pare e
investigue antes de medir.

## 3. Rodar os experimentos

Feche navegador, IDE etc., deixe o notebook na tomada e:

```bash
sudo cpupower frequency-set -g performance   # opcional, mas recomendado
./rodar_experimentos.sh                      # ~10 a 30 min, depende da máquina
python3 analisar.py
```

Para um teste rápido: `REPS=5 TAMANHOS="64 256 1024" ./rodar_experimentos.sh`.

## 4. Overleaf

1. No Overleaf, *New Project → Templates*, procure **SBC Conferences Template** e
   crie o projeto (ele traz `sbc-template.sty` e `sbc.bst`).
2. Suba `main.tex`, `referencias.bib` e as pastas `figuras/` e `resultados/`
   (substitua o `.tex` de exemplo do template).
3. Adicione o professor como colaborador.
4. Preencha a Tabela 1 com `resultados/ambiente.txt`, os autores/instituição e
   os trechos marcados com *[comentar...]* / *[citar a razão]*.

Sempre que rodar de novo, suba de novo `figuras/` e `resultados/`: o texto se
atualiza sozinho.
