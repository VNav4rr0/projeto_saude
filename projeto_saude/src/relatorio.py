
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pseudoanalise import HipotesePseudoanalise, formatar_pseudoanalise_texto

LINHA = "=" * 70


def _secao(titulo: str) -> str:
    return f"\n{LINHA}\n{titulo}\n{LINHA}\n"


def gerar_relatorio(
    caminho_saida: Path,
    n_registros: int,
    estatisticas: pd.DataFrame,
    distribuicoes: dict[str, pd.DataFrame],
    analise_diagnostico: pd.DataFrame,
    matriz_corr: pd.DataFrame,
    estatisticas_protegidas: dict[str, tuple[float, float]],
    epsilon: float,
    hipoteses: list[HipotesePseudoanalise],
    arquivos_graficos: list[str],
) -> None:
    """Gera o arquivo `relatorio.txt` com o resumo completo da execução.

    `estatisticas_protegidas` é um dicionário no formato
    {nome_da_estatistica: (valor_original, valor_com_ruido)}.
    """
    partes = []

    partes.append(LINHA)
    partes.append("RELATÓRIO FINAL — PROJETO ACADÊMICO DE ANÁLISE E PROTEÇÃO")
    partes.append("DE DADOS SINTÉTICOS DE SAÚDE")
    partes.append(LINHA)
    partes.append(
        "\nAVISO: Todos os dados utilizados neste projeto são 100% SINTÉTICOS e "
        "FICTÍCIOS, gerados artificialmente para fins acadêmicos e educacionais. "
        "Não há qualquer informação de pacientes reais. Nenhum resultado deste "
        "relatório deve ser interpretado como diagnóstico médico, recomendação "
        "clínica ou conclusão sobre pessoas reais."
    )

    partes.append(_secao("1. RESUMO DO CONJUNTO DE DADOS"))
    partes.append(f"Quantidade total de registros sintéticos: {n_registros}")
    partes.append(
        "Variáveis: idade (ou faixa etária, na versão protegida), sexo, diagnóstico, "
        "pressão sistólica, pressão diastólica, glicemia, IMC, quantidade de "
        "internações e classificação de risco."
    )

    partes.append(_secao("2. ESTATÍSTICAS DESCRITIVAS (DADOS ORIGINAIS)"))
    partes.append(estatisticas.to_string())

    partes.append(_secao("3. DISTRIBUIÇÕES (FREQUÊNCIA E PERCENTUAL)"))
    for nome, tabela in distribuicoes.items():
        partes.append(f"\n--- Distribuição por {nome} ---")
        partes.append(tabela.to_string())

    partes.append(_secao("4. ANÁLISE POR DIAGNÓSTICO"))
    partes.append(analise_diagnostico.to_string())

    partes.append(_secao("5. MATRIZ DE CORRELAÇÃO"))
    partes.append(matriz_corr.to_string())
    partes.append(
        "\nObservação: valores próximos de 1 ou -1 indicam correlação linear forte "
        "(positiva ou negativa, respectivamente); valores próximos de 0 indicam "
        "pouca ou nenhuma correlação linear. Correlação não implica causalidade."
    )

    partes.append(_secao(f"6. ESTATÍSTICAS PROTEGIDAS COM RUÍDO DE LAPLACE (EPSILON = {epsilon})"))
    partes.append(
        "Demonstração de privacidade diferencial: cada estatística agregada abaixo "
        "recebeu ruído aleatório de Laplace antes de ser divulgada. Valores MENORES "
        "de epsilon aumentam a privacidade e o ruído; valores MAIORES reduzem o "
        "ruído, mas também reduzem a proteção. A escolha real de epsilon em um "
        "sistema de produção exigiria uma análise formal de sensibilidade, "
        "mecanismos de composição e orçamento de privacidade, o que está fora do "
        "escopo deste projeto acadêmico.\n"
    )
    for nome_estatistica, (original, com_ruido) in estatisticas_protegidas.items():
        partes.append(f"  - {nome_estatistica}: original = {original:.2f} | com ruído = {com_ruido:.2f}")

    partes.append(_secao("7. PSEUDOANÁLISE (HIPÓTESES EXPLORATÓRIAS)"))
    partes.append(formatar_pseudoanalise_texto(hipoteses))

    partes.append(_secao("8. GRÁFICOS GERADOS"))
    for nome_arquivo in arquivos_graficos:
        partes.append(f"  - resultados/{nome_arquivo}")

    partes.append(_secao("9. LIMITAÇÕES DO PROJETO"))
    partes.append(
        "- Todos os dados são sintéticos; as relações estatísticas foram criadas\n"
        "  artificialmente para fins didáticos e não representam conhecimento médico.\n"
        "- A remoção de identificadores, a generalização de idade em faixas e o\n"
        "  arredondamento de variáveis numéricas são técnicas de desidentificação,\n"
        "  mas não garantem anonimato absoluto nem eliminam o risco de\n"
        "  reidentificação (por exemplo, via ataques de vinculação de dados).\n"
        "- A demonstração de privacidade diferencial usa um valor fixo de epsilon\n"
        "  e uma sensibilidade simplificada, apenas para fins ilustrativos; um\n"
        "  sistema real exigiria análise formal de sensibilidade, mecanismos de\n"
        "  composição de consultas e gestão de orçamento de privacidade.\n"
        "- A pseudoanálise é heurística e exploratória: não realiza testes de\n"
        "  significância estatística formal, não prova causalidade e não deve ser\n"
        "  usada para qualquer finalidade clínica ou decisão real.\n"
        "- Este projeto tem finalidade exclusivamente acadêmica e educacional."
    )

    conteudo = "\n".join(partes)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_saida.write_text(conteudo, encoding="utf-8-sig")
