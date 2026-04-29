"""
Exemplo 4.1 – Determinação de dias disponíveis para operações mecanizadas

Entrada esperada em CSV: 'dados.csv'
Data;Temperatura_C;Umidade_relativa;Precipitacao_mm;ET0_mm_dia

Observação:
- A coluna ET0_mm_dia pode representar ET0 fornecida pela estação meteorológica.
- O separador pode ser ponto e vírgula (;).
- Valores decimais podem usar vírgula ou ponto.

Saídas geradas:
1) dias_disponiveis_diario.csv
2) resumo_mensal.csv
3) matriz_markov.csv
4) sequencias_operacionais.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DIAS_SEMANA = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo",
}

MESES = {
    1: "Jan",
    2: "Fev",
    3: "Mar",
    4: "Abr",
    5: "Mai",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Set",
    10: "Out",
    11: "Nov",
    12: "Dez",
}


def numero(valor):
    return pd.to_numeric(
        str(valor).replace(".", "").replace(",", ".").strip(),
        errors="coerce"
    )


def ler_csv(caminho):
    codificacoes = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    for enc in codificacoes:
        try:
            df = pd.read_csv(caminho, sep=None, engine="python", encoding=enc)
            print(f"Arquivo lido com codificação: {enc}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Não foi possível ler o arquivo CSV com as codificações testadas.")

    df.columns = [c.strip() for c in df.columns]

    colunas_obrigatorias = [
        "Data",
        "Temperatura_C",
        "Umidade_relativa",
        "Precipitacao_mm",
        "ET0_mm_dia",
    ]

    faltantes = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {faltantes}")

    dados = pd.DataFrame()
    dados["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    dados["Temperatura_C"] = df["Temperatura_C"].apply(numero)
    dados["Umidade_relativa_%"] = df["Umidade_relativa"].apply(numero)
    dados["Precipitacao_mm"] = df["Precipitacao_mm"].apply(numero)
    dados["ET0_mm_dia"] = df["ET0_mm_dia"].apply(numero)

    dados = dados.dropna()
    dados = dados.sort_values("Data").reset_index(drop=True)
    return dados


def classificar_dias(dados, cc, pmp, densidade_aparente, profundidade_cm, limite_umidade=0.90):
    cad = 0.10 * (cc - pmp) * densidade_aparente * profundidade_cm
    lad_anterior = cad
    precipitacao_anterior = 0.0

    resultados = []

    for _, linha in dados.iterrows():
        data = linha["Data"]
        precipitacao = float(linha["Precipitacao_mm"])
        et0 = float(linha["ET0_mm_dia"])

        if cad <= 0:
            raise ValueError("CAD inválida. Verifique CC, PMP, densidade e profundidade.")

        k = np.log(lad_anterior + 1.0) / np.log(cad + 1.0)
        etr = k * et0

        lad = lad_anterior + precipitacao - etr

        if lad > cad:
            lad = cad

        if lad < 0:
            lad = 0

        condicao = "Dia Bom"
        disponivel = 1

        dia_ruim = (
            (precipitacao > 5.0)
            or (lad > limite_umidade * cad)
            or ((precipitacao_anterior > 2.0) and (precipitacao > 0.2))
            or (precipitacao_anterior > 10.0)
        )

        if dia_ruim:
            condicao = "Dia Ruim"
            disponivel = 0

        resultados.append({
            "Data": data.date().isoformat(),
            "Dia_semana": DIAS_SEMANA[data.weekday()],
            "Temperatura_C": linha["Temperatura_C"],
            "Umidade_relativa_%": linha["Umidade_relativa_%"],
            "Precipitacao_mm": precipitacao,
            "ET0_mm_dia": et0,
            "K": k,
            "ETr_mm_dia": etr,
            "LAD_mm": lad,
            "CAD_mm": cad,
            "Condicao": condicao,
            "Disponivel": disponivel,
        })

        lad_anterior = lad
        precipitacao_anterior = precipitacao

    return pd.DataFrame(resultados)


def resumo_mensal(resultado_diario):
    df = resultado_diario.copy()
    df["Data"] = pd.to_datetime(df["Data"])
    df["Mes_num"] = df["Data"].dt.month
    df["Mes"] = df["Mes_num"].map(MESES)

    resumo = df.groupby(["Mes_num", "Mes"]).agg(
        Dias_analisados=("Disponivel", "count"),
        Dias_bons=("Disponivel", "sum"),
    ).reset_index()

    resumo["Dias_ruins"] = resumo["Dias_analisados"] - resumo["Dias_bons"]
    resumo["P_Dia_Bom"] = resumo["Dias_bons"] / resumo["Dias_analisados"]
    resumo["P_Dia_Ruim"] = resumo["Dias_ruins"] / resumo["Dias_analisados"]

    return resumo


def matriz_markov(resultado_diario):
    estados = resultado_diario["Disponivel"].to_numpy()

    contagens = {
        ("Dia Bom", "Dia Bom"): 0,
        ("Dia Bom", "Dia Ruim"): 0,
        ("Dia Ruim", "Dia Bom"): 0,
        ("Dia Ruim", "Dia Ruim"): 0,
    }

    for anterior, atual in zip(estados[:-1], estados[1:]):
        estado_anterior = "Dia Bom" if anterior == 1 else "Dia Ruim"
        estado_atual = "Dia Bom" if atual == 1 else "Dia Ruim"
        contagens[(estado_anterior, estado_atual)] += 1

    linhas = []

    for estado_anterior in ["Dia Bom", "Dia Ruim"]:
        total = (
            contagens[(estado_anterior, "Dia Bom")]
            + contagens[(estado_anterior, "Dia Ruim")]
        )

        linhas.append({
            "Estado_anterior": estado_anterior,
            "N_Dia_Bom": contagens[(estado_anterior, "Dia Bom")],
            "N_Dia_Ruim": contagens[(estado_anterior, "Dia Ruim")],
            "P_Dia_Bom": contagens[(estado_anterior, "Dia Bom")] / total if total > 0 else np.nan,
            "P_Dia_Ruim": contagens[(estado_anterior, "Dia Ruim")] / total if total > 0 else np.nan,
        })

    return pd.DataFrame(linhas)


def sequencias_operacionais(resultado_diario):
    estados = resultado_diario["Disponivel"].to_list()
    datas = resultado_diario["Data"].to_list()

    if not estados:
        return pd.DataFrame(columns=["Condicao", "Data_inicio", "Data_fim", "Duracao_dias"])

    sequencias = []
    estado_atual = estados[0]
    inicio = 0

    for i in range(1, len(estados) + 1):
        fim_da_serie = i == len(estados)
        mudou_estado = not fim_da_serie and estados[i] != estado_atual

        if fim_da_serie or mudou_estado:
            sequencias.append({
                "Condicao": "Dia Bom" if estado_atual == 1 else "Dia Ruim",
                "Data_inicio": datas[inicio],
                "Data_fim": datas[i - 1],
                "Duracao_dias": i - inicio,
            })

            if not fim_da_serie:
                estado_atual = estados[i]
                inicio = i

    return pd.DataFrame(sequencias)


def main():
# ===== EXECUÇÃO DIRETA (Spyder / Aula) =====

    arquivo_csv = "dados.csv"               # ou caminho completo
    pasta_saida = "saida_dias_disponiveis"

# parâmetros (iguais da sua planilha)
    cc = 35.8
    pmp = 20.8
    dap = 1.20
    z = 15.0
    limite = 0.90
    
    dados = ler_csv(arquivo_csv)
    
    diario = classificar_dias(
        dados,
        cc=cc,
        pmp=pmp,
        densidade_aparente=dap,
        profundidade_cm=z,
        limite_umidade=limite,
        )

    mensal = resumo_mensal(diario)
    markov = matriz_markov(diario)
    sequencias = sequencias_operacionais(diario)
    
    Path(pasta_saida).mkdir(exist_ok=True)

    diario.to_csv(f"{pasta_saida}/dias_disponiveis_diario.csv", sep=";", decimal=",", index=False)
    mensal.to_csv(f"{pasta_saida}/resumo_mensal.csv", sep=";", decimal=",", index=False)
    markov.to_csv(f"{pasta_saida}/matriz_markov.csv", sep=";", decimal=",", index=False)
    sequencias.to_csv(f"{pasta_saida}/sequencias_operacionais.csv", sep=";", decimal=",", index=False)
    
    print("Processamento concluído.")
    print(f"Arquivos salvos em: {pasta_saida}")
    
# ============================================================
# GRÁFICOS
# ============================================================

# Preparação dos dados para os gráficos
    diario_graf = diario.copy()
    diario_graf["Data"] = pd.to_datetime(diario_graf["Data"])
    diario_graf["Mes_num"] = diario_graf["Data"].dt.month
    
    prec_mensal = diario_graf.groupby("Mes_num", as_index=False)["Precipitacao_mm"].sum()
    prec_mensal = prec_mensal.rename(columns={"Precipitacao_mm": "Precipitacao_mensal_mm"})
    
    graf = mensal.merge(prec_mensal, on="Mes_num", how="left")
    graf = graf.sort_values("Mes_num")
    
    # ------------------------------------------------------------
    # Figura 4.7 – Dias disponíveis por mês
    # ------------------------------------------------------------
    
    plt.figure(figsize=(10, 6))
    plt.bar(graf["Mes"], graf["Dias_bons"])
    plt.xlabel("Mês")
    plt.ylabel("Dias disponíveis")
    plt.title("Dias disponíveis para operações mecanizadas por mês")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/Figura_4_7_dias_disponiveis_por_mes.png", dpi=300)
    plt.show()
    
    # ------------------------------------------------------------
# Figura 4.8 – Precipitação mensal vs probabilidade
    # ------------------------------------------------------------
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(graf["Mes"], graf["Precipitacao_mensal_mm"], alpha=0.7)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Precipitação mensal (mm)")
     
    ax2 = ax1.twinx()
    ax2.plot(graf["Mes"], graf["P_Dia_Bom"], marker="o")
    ax2.set_ylabel("Probabilidade de dia disponível")
        
    plt.title("Precipitação mensal e probabilidade de dias disponíveis")
    fig.tight_layout()
    plt.savefig(f"{pasta_saida}/Figura_4_8_precipitacao_probabilidade.png", dpi=300)
    plt.show()
        
    print("Gráficos gerados com sucesso.")
    

if __name__ == "__main__":
    main()
