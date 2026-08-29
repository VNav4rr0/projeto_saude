
from __future__ import annotations

import numpy as np
import pandas as pd

# Limites das faixas etárias utilizadas na generalização.
FAIXAS_ETARIAS = [
    (18, 29, "18-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 69, "60-69"),
    (70, 79, "70-79"),
    (80, 200, "80+"),
]


def generalizar_idade(idade: int) -> str:
    """Converte uma idade exata em uma faixa etária generalizada.

    A generalização em faixas é uma técnica clássica de proteção de dados:
    reduz a granularidade da informação (e, portanto, o poder de
    identificação individual), mas não impede reidentificação por si só,
    especialmente quando combinada com outras variáveis.
    """
    for minimo, maximo, rotulo in FAIXAS_ETARIAS:
        if minimo <= idade <= maximo:
            return rotulo
    return "Desconhecida"


def arredondar_para_multiplo(valor: float, multiplo: int) -> int:
    """Arredonda `valor` para o múltiplo mais próximo de `multiplo`.

    Usado para generalizar variáveis numéricas contínuas (glicemia, pressão)
    reduzindo sua precisão original, o que dificulta (mas não impede) a
    identificação de um indivíduo específico por um valor muito exato.
    """
    return int(round(valor / multiplo) * multiplo)


def proteger_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Gera a versão PROTEGIDA/GENERALIZADA do conjunto de dados.

    Passos aplicados:
        1. Remoção do identificador individual (`id_sintetico`).
        2. Generalização da idade exata em faixas etárias.
        3. Arredondamento/generalização de glicemia, pressão arterial e IMC.

    Retorna um novo DataFrame; o DataFrame original não é modificado.
    """
    df_protegido = df.copy()

    # 1) Remoção do identificador individual.
    if "id_sintetico" in df_protegido.columns:
        df_protegido = df_protegido.drop(columns=["id_sintetico"])

    # 2) Generalização da idade em faixas etárias (a idade exata é removida).
    df_protegido["faixa_etaria"] = df_protegido["idade"].apply(generalizar_idade)
    df_protegido = df_protegido.drop(columns=["idade"])

    # 3) Generalização/arredondamento de variáveis numéricas sensíveis.
    #    Exemplo: 118 -> 120 (glicemia), 31.4 -> 31 (IMC), 168 -> 170 (pressão).
    df_protegido["glicemia"] = df_protegido["glicemia"].apply(lambda v: arredondar_para_multiplo(v, 5))
    df_protegido["pressao_sistolica"] = df_protegido["pressao_sistolica"].apply(
        lambda v: arredondar_para_multiplo(v, 5)
    )
    df_protegido["pressao_diastolica"] = df_protegido["pressao_diastolica"].apply(
        lambda v: arredondar_para_multiplo(v, 5)
    )
    df_protegido["imc"] = df_protegido["imc"].round(0).astype(int)

    # Reorganiza colunas para ficar mais legível.
    colunas_ordenadas = [
        "faixa_etaria",
        "sexo",
        "diagnostico",
        "pressao_sistolica",
        "pressao_diastolica",
        "glicemia",
        "imc",
        "qtd_internacoes",
        "classificacao_risco",
    ]
    df_protegido = df_protegido[colunas_ordenadas]
    return df_protegido


def ruido_laplace(valor: float, sensibilidade: float, epsilon: float,
                   rng: np.random.Generator | None = None) -> float:
    """Aplica ruído de Laplace a um valor agregado, seguindo o mecanismo
    clássico de privacidade diferencial: Laplace(0, sensibilidade / epsilon).

    Parâmetros
    ----------
    valor : float
        Estatística agregada original (ex.: uma média calculada sobre o
        conjunto de dados).
    sensibilidade : float
        Sensibilidade da consulta — o quanto o resultado pode mudar, no pior
        caso, se um único registro do conjunto de dados for alterado.
        Definir a sensibilidade corretamente é uma etapa não trivial e, em
        projetos reais, exige uma análise formal.
    epsilon : float
        Parâmetro de privacidade (orçamento de privacidade). Valores
        MENORES de epsilon normalmente oferecem MAIOR privacidade, à custa
        de MAIS ruído (menor utilidade). Valores MAIORES de epsilon
        oferecem MENOS ruído (mais utilidade), mas MENOR proteção de
        privacidade. A escolha de um valor "correto" de epsilon depende do
        contexto, da sensibilidade real da consulta e de um orçamento de
        privacidade acumulado ao longo de múltiplas consultas — este projeto
        usa um valor fixo (EPSILON = 1.0) apenas para fins didáticos.
    rng : numpy.random.Generator, opcional
        Gerador aleatório a utilizar. Se None, cria um novo gerador não
        determinístico. Passe um gerador com semente fixa para resultados
        reprodutíveis.

    Retorna
    -------
    float
        Valor original acrescido de ruído aleatório de Laplace.
    """
    if epsilon <= 0:
        raise ValueError("epsilon deve ser um número positivo.")
    if sensibilidade <= 0:
        raise ValueError("sensibilidade deve ser um número positivo.")

    gerador = rng if rng is not None else np.random.default_rng()
    escala = sensibilidade / epsilon
    ruido = gerador.laplace(loc=0.0, scale=escala)
    return valor + ruido
