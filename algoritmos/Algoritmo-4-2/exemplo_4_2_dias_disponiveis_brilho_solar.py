"""
Algoritmo 4.2 – Determinação de dias disponíveis para operações mecanizadas
usando duração do brilho solar

Entrada esperada em CSV: 'dados_brilho_solar.csv'
Data;Temperatura_C;Umidade_relativa;Precipitacao_mm;Brilho_solar_h

Observação:
- A coluna Brilho_solar_h representa a duração do brilho solar diário (n), em horas.
- O programa calcula ET0/ETp pelo método de radiação.
- O separador pode ser ponto e vírgula (;).
- Valores decimais podem usar vírgula ou ponto.

Saídas geradas:
1) dias_disponiveis_diario.csv
2) resumo_mensal.csv
3) matriz_markov.csv
4) sequencias_operacionais.csv
5) gráficos em PNG
"""

from pathlib import Path
import math
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
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
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
        raise RuntimeError("Não foi possível ler o arquivo CSV com as codificações testadas.")

    df.columns = [c.strip() for c in df.columns]

    colunas_obrigatorias = [
        "Data",
        "Temperatura_C",
        "Umidade_relativa",
        "Precipitacao_mm",
        "Brilho_solar_h",
    ]

    faltantes = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {faltantes}")

    dados = pd.DataFrame()
    dados["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    dados["Temperatura_C"] = df["Temperatura_C"].apply(numero)
    dados["Umidade_relativa_%"] = df["Umidade_relativa"].apply(numero)
    dados["Precipitacao_mm"] = df["Precipitacao_mm"].apply(numero)
    dados["Brilho_solar_h"] = df["Brilho_solar_h"].apply(numero)

    dados = dados.dropna()
    dados = dados.sort_values("Data").reset_index(drop=True)
    return dados


def pvs(T):
    """Pressão de vapor de saturação em função da temperatura do ar."""
    Tk = T + 273.16
    return 51.715 * math.exp(51.594 - 6834 / Tk - 5.169 * math.log(Tk))


def tempum(T, ur, patm):
    """Estima a temperatura de bulbo úmido por busca iterativa."""
    if ur >= 99.999:
        return T

    pvsbs = pvs(T)
    pv = ur * pvsbs / 100.0
    rm = 0.622 * pv / (patm - pv)
    entalpia = 0.24 * T + (597.6 + 0.45 * T) * rm

    t_baixo = -50.0
    t_alto = T

    for _ in range(100):
        tbu = (t_baixo + t_alto) / 2.0
        pvsbu = pvs(tbu)
        rmu = (entalpia - 0.24 * tbu) / (597.6 + 0.45 * tbu)
        ur_calculada = patm * rmu / (pvsbu * (0.622 + rmu))

        if ur_calculada > 1.0:
            t_baixo = tbu
        elif ur_calculada < 0.99:
            t_alto = tbu
        else:
            return tbu

    return (t_baixo + t_alto) / 2.0


def calcular_et0_radiacao(data, T, ur, n, latitude_graus, patm, hemisferio="S"):
    """Calcula ET0/ETp pelo método de radiação usando brilho solar diário."""
    nd = int(pd.Timestamp(data).dayofyear)

    lat = abs(latitude_graus)
    if hemisferio.upper() == "S":
        lat = -lat

    phi = math.radians(lat)

    delta = 23.45 * math.sin((2 * math.pi / 365.0) * (284 + nd)) * math.pi / 180.0

    x = -math.tan(delta) * math.tan(phi)
    x = min(1.0, max(-1.0, x))
    H = math.acos(x)

    N = 24.0 * H / math.pi

    eps = 2.0 * math.pi * (nd - 1) / 365.0
    dmd2 = (
        1.000110
        + 0.034221 * math.cos(eps)
        + 0.001280 * math.sin(eps)
        + 0.000719 * math.cos(2 * eps)
        + 0.000077 * math.sin(2 * eps)
    )

    Q0 = (1440.0 / math.pi) * 1.94 * dmd2 * (
        H * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * math.sin(H)
    )

    pvsbs = pvs(T)
    Tbu = tempum(T, ur, patm)
    pvsbu = pvs(Tbu)
    pv = ur * pvsbs / 100.0

    W = 1.0 if abs(pvsbs - pv) < 1e-12 else (pvsbs - pvsbu) / (pvsbs - pv)
    Ra = Q0 / 59.0

    et0 = W * (0.29 * math.cos(phi) + 0.52 * (n / N)) * Ra
    return max(0.0, et0)


def classificar_dias(
    dados,
    cc,
    pmp,
    densidade_aparente,
    profundidade_cm,
    latitude_graus,
    patm,
    hemisferio="S",
    limite_umidade=0.90,
):
    cad = 0.10 * (cc - pmp) * densidade_aparente * profundidade_cm
    lad_anterior = cad
    precipitacao_anterior = 0.0

    resultados = []

    for _, linha in dados.iterrows():
        data = linha["Data"]
        T = float(linha["Temperatura_C"])
        ur = float(linha["Umidade_relativa_%"])
        precipitacao = float(linha["Precipitacao_mm"])
        brilho_solar = float(linha["Brilho_solar_h"])

        if cad <= 0:
            raise ValueError("CAD inválida. Verifique CC, PMP, densidade e profundidade.")

        et0 = calcular_et0_radiacao(
            data=data,
            T=T,
            ur=ur,
            n=brilho_solar,
            latitude_graus=latitude_graus,
            patm=patm,
            hemisferio=hemisferio,
        )

        k = np.log(lad_anterior + 1.0) / np.log(cad + 1.0)
        etr = k * et0

        lad = lad_anterior + precipitacao - etr
        lad = min(lad, cad)
        lad = max(lad, 0)

        dia_ruim = (
            (precipitacao > 5.0)
            or (lad > limite_umidade * cad)
            or ((precipitacao_anterior > 2.0) and (precipitacao > 0.2))
            or (precipitacao_anterior > 10.0)
        )

        condicao = "Dia Ruim" if dia_ruim else "Dia Bom"
        disponivel = 0 if dia_ruim else 1

        resultados.append({
            "Data": data.date().isoformat(),
            "Dia_semana": DIAS_SEMANA[data.weekday()],
            "Temperatura_C": T,
            "Umidade_relativa_%": ur,
            "Precipitacao_mm": precipitacao,
            "Brilho_solar_h": brilho_solar,
            "ET0_calculada_mm_dia": et0,
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
        total = contagens[(estado_anterior, "Dia Bom")] + contagens[(estado_anterior, "Dia Ruim")]
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


def gerar_graficos(diario, mensal, pasta_saida):
    diario_graf = diario.copy()
    diario_graf["Data"] = pd.to_datetime(diario_graf["Data"])
    diario_graf["Mes_num"] = diario_graf["Data"].dt.month

    prec_mensal = diario_graf.groupby("Mes_num", as_index=False)["Precipitacao_mm"].sum()
    prec_mensal = prec_mensal.rename(columns={"Precipitacao_mm": "Precipitacao_mensal_mm"})

    graf = mensal.merge(prec_mensal, on="Mes_num", how="left")
    graf = graf.sort_values("Mes_num")

    plt.figure(figsize=(10, 6))
    plt.bar(graf["Mes"], graf["Dias_bons"])
    plt.xlabel("Mês")
    plt.ylabel("Dias disponíveis")
    plt.title("Dias disponíveis para operações mecanizadas por mês")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/Figura_4_9_dias_disponiveis_por_mes.png", dpi=300)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(graf["Mes"], graf["Precipitacao_mensal_mm"], alpha=0.7)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Precipitação mensal (mm)")

    ax2 = ax1.twinx()
    ax2.plot(graf["Mes"], graf["P_Dia_Bom"], marker="o")
    ax2.set_ylabel("Probabilidade de dia disponível")

    plt.title("Precipitação mensal e probabilidade de dias disponíveis")
    fig.tight_layout()
    plt.savefig(f"{pasta_saida}/Figura_4_10_precipitacao_probabilidade.png", dpi=300)
    plt.show()


def main():
    # ===== EXECUÇÃO DIRETA (Spyder / Aula) =====

    arquivo_csv = "dados_brilho_solar_exemplo.csv"
    pasta_saida = "saida_dias_disponiveis_brilho_solar"

    # Dados do solo
    cc = 31.0
    pmp = 22.9
    dap = 0.85
    z = 15.0
    limite = 0.90

    # Condições locais
    latitude_graus = 19 + 45 / 60      # 19°45'
    hemisferio = "S"
    patm = 720.0                       # mmHg

    dados = ler_csv(arquivo_csv)

    diario = classificar_dias(
        dados,
        cc=cc,
        pmp=pmp,
        densidade_aparente=dap,
        profundidade_cm=z,
        latitude_graus=latitude_graus,
        patm=patm,
        hemisferio=hemisferio,
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

    gerar_graficos(diario, mensal, pasta_saida)

    print("Processamento concluído.")
    print(f"Arquivos salvos em: {pasta_saida}")


if __name__ == "__main__":
    main()
