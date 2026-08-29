

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analise import (
    analise_por_diagnostico,
    estatisticas_descritivas,
    gerar_todas_distribuicoes,
    matriz_correlacao,
)
from src.gerar_dados import gerar_dados_sinteticos
from src.protecao import proteger_dados, ruido_laplace
from src.pseudoanalise import gerar_pseudoanalise
from src.relatorio import gerar_relatorio
from src.visualizacao import gerar_todos_os_graficos

# ----------------------------------------------------------------------
# Configurações gerais do projeto
# ----------------------------------------------------------------------
N_REGISTROS = 1000
SEMENTE = 42
EPSILON = 1.0  # parâmetro de privacidade diferencial (ver README.md)

PASTA_RAIZ = Path(__file__).resolve().parent
PASTA_DADOS = PASTA_RAIZ / "dados"
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"

CAMINHO_DADOS_ORIGINAIS = PASTA_DADOS / "pacientes_sinteticos.csv"
CAMINHO_DADOS_PROTEGIDOS = PASTA_RESULTADOS / "dados_protegidos.csv"
CAMINHO_RELATORIO = PASTA_RESULTADOS / "relatorio.txt"


def _imprimir_etapa(numero: int, descricao: str) -> None:
    print(f"\n[Etapa {numero}] {descricao}")


def main() -> None:
    """Executa o pipeline completo do projeto, do início ao fim."""
    try:
        linha_separadora = "=" * 70
        print(linha_separadora)
        print("PROJETO ACADÊMICO — ANÁLISE E PROTEÇÃO DE DADOS SINTÉTICOS DE SAÚDE")
        print(linha_separadora)
        print(
            "AVISO: todos os dados são sintéticos e fictícios. Uso exclusivamente\n"
            "acadêmico/educacional. Nenhum resultado representa diagnóstico ou\n"
            "recomendação médica real."
        )

        # Garante que as pastas de saída existam.
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

        # --- Etapa 1: geração dos dados sintéticos --------------------------
        _imprimir_etapa(1, f"Gerando {N_REGISTROS} registros sintéticos (semente={SEMENTE})...")
        df = gerar_dados_sinteticos(n_registros=N_REGISTROS, semente=SEMENTE)
        df.to_csv(CAMINHO_DADOS_ORIGINAIS, index=False, encoding="utf-8-sig")
        print(f"  -> Arquivo criado: {CAMINHO_DADOS_ORIGINAIS}")

        # --- Etapa 2: proteção / anonimização / generalização ----------------
        _imprimir_etapa(2, "Aplicando proteção/anonimização/generalização aos dados...")
        df_protegido = proteger_dados(df)
        df_protegido.to_csv(CAMINHO_DADOS_PROTEGIDOS, index=False, encoding="utf-8-sig")
        print(f"  -> Arquivo criado: {CAMINHO_DADOS_PROTEGIDOS}")

        # --- Etapa 3: estatísticas descritivas --------------------------------
        _imprimir_etapa(3, "Calculando estatísticas descritivas...")
        estatisticas = estatisticas_descritivas(df)
        print(estatisticas.to_string())

        # --- Etapa 4: distribuições --------------------------------------------
        _imprimir_etapa(4, "Calculando distribuições por categoria...")
        distribuicoes = gerar_todas_distribuicoes(df_protegido)
        for nome, tabela in distribuicoes.items():
            print(f"\nDistribuição por {nome}:")
            print(tabela.to_string())

        # --- Etapa 5: análise por diagnóstico -----------------------------------
        _imprimir_etapa(5, "Realizando análise agrupada por diagnóstico...")
        analise_diag = analise_por_diagnostico(df)
        print(analise_diag.to_string())

        # --- Etapa 6: matriz de correlação --------------------------------------
        _imprimir_etapa(6, "Calculando matriz de correlação...")
        matriz_corr = matriz_correlacao(df)
        print(matriz_corr.to_string())

        # --- Etapa 7: privacidade diferencial (ruído de Laplace) ----------------
        _imprimir_etapa(7, f"Aplicando ruído de Laplace (epsilon={EPSILON}) a estatísticas agregadas...")
        rng_privacidade = np.random.default_rng(SEMENTE)
        # Sensibilidade aproximada: estimativa didática do impacto máximo de um
        # único registro sobre a média, considerando o tamanho da amostra.
        estatisticas_protegidas = {}
        for coluna, sensibilidade in [
            ("glicemia", 400 / N_REGISTROS),
            ("imc", 50 / N_REGISTROS),
            ("pressao_sistolica", 220 / N_REGISTROS),
            ("qtd_internacoes", 10 / N_REGISTROS),
        ]:
            valor_original = float(df[coluna].mean())
            valor_com_ruido = ruido_laplace(valor_original, sensibilidade, EPSILON, rng=rng_privacidade)
            estatisticas_protegidas[f"media_{coluna}"] = (valor_original, valor_com_ruido)
            print(f"  -> media_{coluna}: original={valor_original:.2f} | com ruído={valor_com_ruido:.2f}")

        # --- Etapa 8: pseudoanálise -----------------------------------------------
        _imprimir_etapa(8, "Executando pseudoanálise (hipóteses exploratórias)...")
        hipoteses = gerar_pseudoanalise(df)
        for hipotese in hipoteses:
            print("\n" + hipotese.formatar())

        # --- Etapa 9: geração dos gráficos ------------------------------------------
        _imprimir_etapa(9, "Gerando gráficos em PNG...")
        arquivos_graficos = gerar_todos_os_graficos(df, matriz_corr, PASTA_RESULTADOS)
        for arquivo in arquivos_graficos:
            print(f"  -> Gráfico criado: resultados/{arquivo}")

        # --- Etapa 10: relatório final -----------------------------------------------
        _imprimir_etapa(10, "Gerando relatório final...")
        gerar_relatorio(
            caminho_saida=CAMINHO_RELATORIO,
            n_registros=N_REGISTROS,
            estatisticas=estatisticas,
            distribuicoes=distribuicoes,
            analise_diagnostico=analise_diag,
            matriz_corr=matriz_corr,
            estatisticas_protegidas=estatisticas_protegidas,
            epsilon=EPSILON,
            hipoteses=hipoteses,
            arquivos_graficos=arquivos_graficos,
        )
        print(f"  -> Relatório criado: {CAMINHO_RELATORIO}")

        print("\n" + "=" * 70)
        print("EXECUÇÃO CONCLUÍDA COM SUCESSO")
        print("=" * 70)
        print("\nArquivos gerados:")
        print(f"  - {CAMINHO_DADOS_ORIGINAIS}")
        print(f"  - {CAMINHO_DADOS_PROTEGIDOS}")
        print(f"  - {CAMINHO_RELATORIO}")
        for arquivo in arquivos_graficos:
            print(f"  - {PASTA_RESULTADOS / arquivo}")

    except Exception as erro:  # tratamento genérico de erros para facilitar diagnóstico
        print("\n" + "!" * 70)
        print("ERRO DURANTE A EXECUÇÃO DO PROJETO")
        print("!" * 70)
        print(f"Detalhes: {erro}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
