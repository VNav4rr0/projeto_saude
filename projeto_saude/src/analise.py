"""
Módulo: analise.py

Responsável pelas análises estatísticas descritivas, distribuições,
análises agrupadas por diagnóstico e matriz de correlação.

Todas as análises operam sobre o DataFrame de dados sintéticos e retornam
estruturas (dicionários / DataFrames) que são usadas pelo módulo de
relatório e pelo módulo de visualização.
"""

from __future__ import annotations

import pandas as pd

COLUNAS_NUMERICAS = [
    "glicemia",
    "imc",
    "pressao_sistolica",
    "pressao_diastolica",
    "qtd_internacoes",
]


def estatisticas_descritivas(df: pd.DataFrame, colunas: list[str] = None) -> pd.DataFrame:
    """Calcula estatísticas descritivas (contagem, média, mediana, desvio
    padrão, variância, mínimo, máximo e quartis) para as colunas numéricas
    informadas.

    Retorna um DataFrame com uma linha por variável e uma coluna por
    estatística, o que facilita tanto a exibição no terminal quanto a
    inclusão no relatório final.
    """
    colunas = colunas or COLUNAS_NUMERICAS
    linhas = []
    for coluna in colunas:
        serie = df[coluna]
        linhas.append(
            {
                "variavel": coluna,
                "contagem": serie.count(),
                "media": serie.mean(),
                "mediana": serie.median(),
                "desvio_padrao": serie.std(),
                "variancia": serie.var(),
                "minimo": serie.min(),
                "maximo": serie.max(),
                "q1_25%": serie.quantile(0.25),
                "q3_75%": serie.quantile(0.75),
            }
        )
    resultado = pd.DataFrame(linhas).set_index("variavel")
    return resultado.round(2)


def distribuicao_percentual(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Retorna a frequência absoluta e o percentual de cada categoria de
    `coluna` (por exemplo: sexo, faixa etária, diagnóstico ou risco).
    """
    contagem = df[coluna].value_counts(dropna=False)
    percentual = df[coluna].value_counts(normalize=True, dropna=False) * 100
    resultado = pd.DataFrame({"quantidade": contagem, "percentual_%": percentual.round(2)})
    resultado.index.name = coluna
    return resultado


def analise_por_diagnostico(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os registros por diagnóstico e calcula métricas médias de
    interesse (glicemia, IMC, pressão sistólica e internações), além da
    quantidade de registros em cada grupo.
    """
    agrupado = df.groupby("diagnostico").agg(
        quantidade_registros=("diagnostico", "count"),
        glicemia_media=("glicemia", "mean"),
        imc_medio=("imc", "mean"),
        pressao_sistolica_media=("pressao_sistolica", "mean"),
        internacoes_media=("qtd_internacoes", "mean"),
    )
    return agrupado.round(2).sort_values("quantidade_registros", ascending=False)


def matriz_correlacao(df: pd.DataFrame, colunas: list[str] = None) -> pd.DataFrame:
    """Calcula a matriz de correlação de Pearson entre as colunas numéricas
    informadas (padrão: glicemia, IMC, pressões e internações).
    """
    colunas = colunas or COLUNAS_NUMERICAS
    return df[colunas].corr().round(3)


def gerar_todas_distribuicoes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Gera as distribuições por sexo, faixa etária (quando disponível),
    classificação de risco e diagnóstico, retornando um dicionário
    {nome_da_distribuicao: DataFrame}.
    """
    distribuicoes = {
        "sexo": distribuicao_percentual(df, "sexo"),
        "classificacao_risco": distribuicao_percentual(df, "classificacao_risco"),
        "diagnostico": distribuicao_percentual(df, "diagnostico"),
    }
    if "faixa_etaria" in df.columns:
        distribuicoes["faixa_etaria"] = distribuicao_percentual(df, "faixa_etaria")
    return distribuicoes
