

from __future__ import annotations

import numpy as np
import pandas as pd

# Lista fixa de diagnósticos fictícios utilizados no projeto.
DIAGNOSTICOS = [
    "Nenhum",
    "Hipertensão",
    "Diabetes tipo 2",
    "Asma",
    "Hipotireoidismo",
]

# Probabilidade de cada diagnóstico aparecer em um registro sintético.
PROBABILIDADES_DIAGNOSTICO = [0.40, 0.20, 0.20, 0.10, 0.10]

SEXOS = ["F", "M"]

NIVEIS_RISCO = ["Baixo", "Moderado", "Alto"]


def _gerar_pressao(rng: np.random.Generator, diagnostico: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gera pressão sistólica e diastólica com relação artificial ao diagnóstico.

    Registros com "Hipertensão" recebem, propositalmente, uma média mais alta
    de pressão arterial, para criar um padrão estatístico demonstrável.
    """
    n = len(diagnostico)
    base_sistolica = rng.normal(loc=115, scale=10, size=n)
    base_diastolica = rng.normal(loc=75, scale=7, size=n)

    ajuste_hipertensao = np.where(diagnostico == "Hipertensão", rng.normal(25, 6, size=n), 0.0)
    ajuste_hipertensao_dia = np.where(diagnostico == "Hipertensão", rng.normal(12, 4, size=n), 0.0)

    sistolica = base_sistolica + ajuste_hipertensao
    diastolica = base_diastolica + ajuste_hipertensao_dia

    # Limites plausíveis apenas para manter os valores em uma faixa razoável.
    sistolica = np.clip(sistolica, 85, 220)
    diastolica = np.clip(diastolica, 50, 140)
    return sistolica, diastolica


def _gerar_glicemia(rng: np.random.Generator, diagnostico: np.ndarray, idade: np.ndarray) -> np.ndarray:
    """Gera glicemia com relação artificial ao diagnóstico "Diabetes tipo 2"
    e um leve efeito artificial da idade, apenas para fins didáticos.
    """
    n = len(diagnostico)
    base = rng.normal(loc=90, scale=10, size=n)
    ajuste_diabetes = np.where(diagnostico == "Diabetes tipo 2", rng.normal(55, 15, size=n), 0.0)
    ajuste_idade = (idade - 40) * 0.15  # tendência artificial leve associada à idade
    glicemia = base + ajuste_diabetes + ajuste_idade
    return np.clip(glicemia, 60, 400)


def _gerar_imc(rng: np.random.Generator, diagnostico: np.ndarray) -> np.ndarray:
    """Gera IMC com leve relação artificial a diagnósticos metabólicos."""
    n = len(diagnostico)
    base = rng.normal(loc=25, scale=4, size=n)
    ajuste = np.where(
        np.isin(diagnostico, ["Diabetes tipo 2", "Hipertensão"]),
        rng.normal(3, 2, size=n),
        0.0,
    )
    imc = base + ajuste
    return np.clip(imc, 15, 50)


def _classificar_risco(idade: np.ndarray, imc: np.ndarray, diagnostico: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Calcula uma classificação de risco artificial a partir de idade, IMC e
    diagnóstico, com um componente aleatório para simular incerteza.

    Esta é uma regra fictícia criada exclusivamente para o projeto, sem
    qualquer validade clínica.
    """
    pontuacao = np.zeros(len(idade))
    pontuacao += (idade >= 60) * 1.5
    pontuacao += (idade >= 75) * 1.0
    pontuacao += (imc >= 30) * 1.5
    pontuacao += np.isin(diagnostico, ["Diabetes tipo 2", "Hipertensão"]) * 1.5
    pontuacao += (diagnostico == "Nenhum") * -1.0
    pontuacao += rng.normal(0, 0.8, size=len(idade))  # ruído para simular incerteza

    risco = np.where(pontuacao >= 3.0, "Alto", np.where(pontuacao >= 1.2, "Moderado", "Baixo"))
    return risco


def _gerar_internacoes(rng: np.random.Generator, risco: np.ndarray) -> np.ndarray:
    """Gera quantidade de internações com relação artificial ao nível de risco:
    quanto maior o risco, maior a média (lambda) da distribuição de Poisson
    utilizada. Isso é uma escolha de design do conjunto sintético, não um
    fato médico.
    """
    lambdas = np.where(risco == "Alto", 2.5, np.where(risco == "Moderado", 1.0, 0.3))
    return rng.poisson(lam=lambdas)


def gerar_dados_sinteticos(n_registros: int = 1000, semente: int = 42) -> pd.DataFrame:
    """Gera um DataFrame com `n_registros` pacientes fictícios.

    Parâmetros
    ----------
    n_registros : int
        Quantidade de registros sintéticos a gerar (padrão: 1000).
    semente : int
        Semente do gerador aleatório, para reprodutibilidade dos resultados.

    Retorna
    -------
    pandas.DataFrame
        DataFrame com os dados sintéticos originais (não protegidos).
    """
    if n_registros <= 0:
        raise ValueError("n_registros deve ser um número inteiro positivo.")

    rng = np.random.default_rng(semente)

    id_sintetico = np.arange(1, n_registros + 1)
    idade = rng.integers(18, 90, size=n_registros)
    sexo = rng.choice(SEXOS, size=n_registros)
    diagnostico = rng.choice(DIAGNOSTICOS, size=n_registros, p=PROBABILIDADES_DIAGNOSTICO)

    pressao_sistolica, pressao_diastolica = _gerar_pressao(rng, diagnostico)
    glicemia = _gerar_glicemia(rng, diagnostico, idade)
    imc = _gerar_imc(rng, diagnostico)
    classificacao_risco = _classificar_risco(idade, imc, diagnostico, rng)
    qtd_internacoes = _gerar_internacoes(rng, classificacao_risco)

    df = pd.DataFrame(
        {
            "id_sintetico": id_sintetico,
            "idade": idade,
            "sexo": sexo,
            "diagnostico": diagnostico,
            "pressao_sistolica": np.round(pressao_sistolica, 1),
            "pressao_diastolica": np.round(pressao_diastolica, 1),
            "glicemia": np.round(glicemia, 1),
            "imc": np.round(imc, 1),
            "qtd_internacoes": qtd_internacoes,
            "classificacao_risco": classificacao_risco,
        }
    )
    return df
