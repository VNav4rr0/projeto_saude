
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend não interativo: evita problemas em ambientes sem GUI

import matplotlib.pyplot as plt
import pandas as pd


def _salvar_figura(fig: plt.Figure, caminho: Path) -> None:
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def grafico_distribuicao_categorica(df: pd.DataFrame, coluna: str, titulo: str, caminho: Path) -> None:
    """Gera um gráfico de barras com a contagem de cada categoria de `coluna`."""
    contagem = df[coluna].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(contagem.index.astype(str), contagem.values, color="#3b6ea5")
    ax.set_title(titulo)
    ax.set_xlabel(coluna)
    ax.set_ylabel("Quantidade de registros")
    ax.tick_params(axis="x", rotation=30)
    _salvar_figura(fig, caminho)


def grafico_histograma(df: pd.DataFrame, coluna: str, titulo: str, caminho: Path, bins: int = 30) -> None:
    """Gera um histograma para uma variável numérica contínua."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[coluna].dropna(), bins=bins, color="#5a9367", edgecolor="black")
    ax.set_title(titulo)
    ax.set_xlabel(coluna)
    ax.set_ylabel("Frequência")
    _salvar_figura(fig, caminho)


def grafico_matriz_correlacao(matriz_corr: pd.DataFrame, caminho: Path) -> None:
    """Gera um heatmap (usando apenas Matplotlib, sem seaborn) da matriz de
    correlação informada.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz_corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(matriz_corr.columns)))
    ax.set_yticks(range(len(matriz_corr.index)))
    ax.set_xticklabels(matriz_corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(matriz_corr.index)

    # Anota cada célula com o valor numérico da correlação.
    for i in range(len(matriz_corr.index)):
        for j in range(len(matriz_corr.columns)):
            valor = matriz_corr.values[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                     color="black", fontsize=9)

    ax.set_title("Matriz de correlação (dados sintéticos)")
    fig.colorbar(im, ax=ax, label="Coeficiente de correlação de Pearson")
    _salvar_figura(fig, caminho)


def gerar_todos_os_graficos(df: pd.DataFrame, matriz_corr: pd.DataFrame, pasta_resultados: Path) -> list[str]:
    """Gera todos os gráficos exigidos pelo projeto e retorna a lista de
    nomes de arquivos criados (relativos à pasta de resultados).
    """
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    arquivos_gerados = []

    graficos = [
        ("distribuicao_diagnosticos.png", lambda caminho: grafico_distribuicao_categorica(
            df, "diagnostico", "Distribuição dos diagnósticos (dados sintéticos)", caminho)),
        ("distribuicao_risco.png", lambda caminho: grafico_distribuicao_categorica(
            df, "classificacao_risco", "Distribuição da classificação de risco (dados sintéticos)", caminho)),
        ("distribuicao_glicemia.png", lambda caminho: grafico_histograma(
            df, "glicemia", "Distribuição da glicemia (dados sintéticos)", caminho)),
        ("distribuicao_imc.png", lambda caminho: grafico_histograma(
            df, "imc", "Distribuição do IMC (dados sintéticos)", caminho)),
        ("matriz_correlacao.png", lambda caminho: grafico_matriz_correlacao(matriz_corr, caminho)),
    ]

    for nome_arquivo, funcao_geradora in graficos:
        caminho = pasta_resultados / nome_arquivo
        funcao_geradora(caminho)
        arquivos_gerados.append(nome_arquivo)

    return arquivos_gerados
