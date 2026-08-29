from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HipotesePseudoanalise:
    """Estrutura de uma hipótese exploratória gerada pela pseudoanálise."""

    observacao: str
    hipotese: str
    confianca: str
    limitacao: str

    def formatar(self) -> str:
        return (
            f"OBSERVAÇÃO: {self.observacao}\n"
            f"HIPÓTESE: {self.hipotese}\n"
            f"CONFIANÇA: {self.confianca}\n"
            f"LIMITAÇÃO: {self.limitacao}\n"
        )


LIMITACAO_PADRAO = (
    "Baseado em dados 100% sintéticos e fictícios, com relações estatísticas "
    "criadas artificialmente para fins didáticos. Não representa evidência "
    "médica real, não prova causalidade e não deve ser usado para qualquer "
    "decisão clínica."
)


def _classificar_confianca(diferenca_relativa: float) -> str:
    """Classifica a "confiança" da hipótese de forma puramente heurística,
    com base no tamanho da diferença relativa observada entre grupos.
    Isso NÃO é um teste estatístico formal de significância.
    """
    diferenca_absoluta = abs(diferenca_relativa)
    if diferenca_absoluta >= 0.30:
        return "Alta (heurística, não é teste estatístico formal)"
    if diferenca_absoluta >= 0.10:
        return "Moderada (heurística, não é teste estatístico formal)"
    return "Baixa (heurística, não é teste estatístico formal)"


def gerar_pseudoanalise(df: pd.DataFrame) -> list[HipotesePseudoanalise]:
    """Percorre os dados e gera uma lista de hipóteses exploratórias com base
    em diferenças observadas entre grupos (diagnóstico e classificação de
    risco). Retorna uma lista de objetos HipotesePseudoanalise.
    """
    hipoteses: list[HipotesePseudoanalise] = []

    media_geral_glicemia = df["glicemia"].mean()
    media_geral_pressao = df["pressao_sistolica"].mean()
    media_geral_internacoes = df["qtd_internacoes"].mean()
    media_geral_imc = df["imc"].mean()

    # --- Glicemia x Diabetes tipo 2 ---
    if "Diabetes tipo 2" in df["diagnostico"].unique():
        media_diabetes = df.loc[df["diagnostico"] == "Diabetes tipo 2", "glicemia"].mean()
        diferenca = (media_diabetes - media_geral_glicemia) / media_geral_glicemia
        hipoteses.append(
            HipotesePseudoanalise(
                observacao=(
                    f"A glicemia média do grupo com diagnóstico 'Diabetes tipo 2' "
                    f"({media_diabetes:.1f}) é diferente da glicemia média geral "
                    f"({media_geral_glicemia:.1f})."
                ),
                hipotese=(
                    "Existe uma associação aparente, nos dados sintéticos, entre o "
                    "diagnóstico 'Diabetes tipo 2' e níveis de glicemia mais elevados."
                ),
                confianca=_classificar_confianca(diferenca),
                limitacao=LIMITACAO_PADRAO,
            )
        )

    # --- Pressão sistólica x Hipertensão ---
    if "Hipertensão" in df["diagnostico"].unique():
        media_hipertensao = df.loc[df["diagnostico"] == "Hipertensão", "pressao_sistolica"].mean()
        diferenca = (media_hipertensao - media_geral_pressao) / media_geral_pressao
        hipoteses.append(
            HipotesePseudoanalise(
                observacao=(
                    f"A pressão sistólica média do grupo com diagnóstico 'Hipertensão' "
                    f"({media_hipertensao:.1f}) é diferente da média geral "
                    f"({media_geral_pressao:.1f})."
                ),
                hipotese=(
                    "Existe uma associação aparente, nos dados sintéticos, entre o "
                    "diagnóstico 'Hipertensão' e níveis mais elevados de pressão sistólica."
                ),
                confianca=_classificar_confianca(diferenca),
                limitacao=LIMITACAO_PADRAO,
            )
        )

    # --- Internações x Classificação de risco ---
    if "Alto" in df["classificacao_risco"].unique():
        media_alto_risco = df.loc[df["classificacao_risco"] == "Alto", "qtd_internacoes"].mean()
        diferenca = (media_alto_risco - media_geral_internacoes) / max(media_geral_internacoes, 1e-6)
        hipoteses.append(
            HipotesePseudoanalise(
                observacao=(
                    f"A quantidade média de internações no grupo classificado como "
                    f"'Alto' risco ({media_alto_risco:.2f}) é diferente da média geral "
                    f"({media_geral_internacoes:.2f})."
                ),
                hipotese=(
                    "Existe uma associação aparente, nos dados sintéticos, entre a "
                    "classificação de risco 'Alto' e uma maior quantidade de internações."
                ),
                confianca=_classificar_confianca(diferenca),
                limitacao=LIMITACAO_PADRAO,
            )
        )

    # --- IMC x Classificação de risco ---
    if "Alto" in df["classificacao_risco"].unique():
        imc_alto_risco = df.loc[df["classificacao_risco"] == "Alto", "imc"].mean()
        diferenca = (imc_alto_risco - media_geral_imc) / media_geral_imc
        hipoteses.append(
            HipotesePseudoanalise(
                observacao=(
                    f"O IMC médio do grupo classificado como 'Alto' risco "
                    f"({imc_alto_risco:.1f}) é diferente do IMC médio geral "
                    f"({media_geral_imc:.1f})."
                ),
                hipotese=(
                    "Existe uma associação aparente, nos dados sintéticos, entre o IMC "
                    "e a classificação de risco atribuída."
                ),
                confianca=_classificar_confianca(diferenca),
                limitacao=LIMITACAO_PADRAO,
            )
        )

    # --- Idade (via faixa etária, se disponível) x Classificação de risco ---
    coluna_idade = "faixa_etaria" if "faixa_etaria" in df.columns else ("idade" if "idade" in df.columns else None)
    if coluna_idade is not None and "Alto" in df["classificacao_risco"].unique():
        if coluna_idade == "idade":
            media_idade_alto = df.loc[df["classificacao_risco"] == "Alto", "idade"].mean()
            media_idade_geral = df["idade"].mean()
            diferenca = (media_idade_alto - media_idade_geral) / media_idade_geral
            observacao = (
                f"A idade média do grupo classificado como 'Alto' risco "
                f"({media_idade_alto:.1f} anos) é diferente da idade média geral "
                f"({media_idade_geral:.1f} anos)."
            )
        else:
            faixa_mais_comum_alto = (
                df.loc[df["classificacao_risco"] == "Alto", "faixa_etaria"].mode().iloc[0]
            )
            faixa_mais_comum_geral = df["faixa_etaria"].mode().iloc[0]
            diferenca = 0.30 if faixa_mais_comum_alto != faixa_mais_comum_geral else 0.0
            observacao = (
                f"A faixa etária mais frequente entre os registros de 'Alto' risco é "
                f"'{faixa_mais_comum_alto}', enquanto a faixa etária mais frequente no "
                f"conjunto geral é '{faixa_mais_comum_geral}'."
            )
        hipoteses.append(
            HipotesePseudoanalise(
                observacao=observacao,
                hipotese=(
                    "Existe uma associação aparente, nos dados sintéticos, entre a idade "
                    "e a classificação de risco atribuída."
                ),
                confianca=_classificar_confianca(diferenca),
                limitacao=LIMITACAO_PADRAO,
            )
        )

    return hipoteses


def formatar_pseudoanalise_texto(hipoteses: list[HipotesePseudoanalise]) -> str:
    """Formata a lista de hipóteses como um texto único, pronto para ser
    exibido no terminal ou incluído no relatório final.
    """
    aviso = (
        "=" * 70 + "\n"
        "PSEUDOANÁLISE EXPLORATÓRIA (NÃO É DIAGNÓSTICO MÉDICO)\n"
        "Correlação/associação estatística NÃO significa causalidade.\n"
        "Todos os padrões abaixo vêm de dados sintéticos e exigem investigação\n"
        "adicional antes de qualquer conclusão.\n" + "=" * 70 + "\n\n"
    )
    corpo = "\n".join(h.formatar() for h in hipoteses)
    return aviso + corpo
