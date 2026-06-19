import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


def analizar_dataset(csv_file):

    # =========================
    # CARPETAS
    # =========================

    os.makedirs("resultados", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)

    # =========================
    # LEER CSV
    # =========================

    df = pd.read_csv(csv_file)

    # =========================
    # ENTROPIAS
    # =========================

    entropias = {}

    for col in df.columns:

        probs = df[col].value_counts(normalize=True)

        entropias[col] = round(
            entropy(probs, base=2),
            6
        )

    entropias_df = pd.DataFrame(
        list(entropias.items()),
        columns=["Variable", "Entropia"]
    )

    entropias_df.to_csv(
        "resultados/entropias.csv",
        index=False
    )

    # =========================
    # MATRIZ MI
    # =========================

    mi_matrix = pd.DataFrame(
        np.zeros((len(df.columns), len(df.columns))),
        index=df.columns,
        columns=df.columns
    )

    for c1 in df.columns:

        for c2 in df.columns:

            mi = mutual_info_score(
                df[c1],
                df[c2]
            )

            mi_matrix.loc[c1, c2] = mi

    mi_matrix.to_csv(
        "resultados/matriz_MI.csv"
    )

    # =========================
    # GRAFO COMPLETO
    # =========================

    G = nx.Graph()

    for nodo in df.columns:
        G.add_node(nodo)

    for i in range(len(df.columns)):

        for j in range(i + 1, len(df.columns)):

            peso = float(
                mi_matrix.iloc[i, j]
            )

            G.add_edge(
                df.columns[i],
                df.columns[j],
                weight=peso
            )

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2500,
        font_size=10
    )

    plt.title(
        "Grafo Completo - Informacion Mutua"
    )

    plt.savefig(
        "static/img/grafo_completo.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # KRUSKAL
    # =========================

    mst_k = nx.maximum_spanning_tree(
        G,
        algorithm="kruskal"
    )

    kruskal_data = []

    peso_kruskal = 0

    for u, v, d in mst_k.edges(data=True):

        peso_kruskal += d["weight"]

        kruskal_data.append([
            u,
            v,
            round(d["weight"], 6)
        ])

    pd.DataFrame(
        kruskal_data,
        columns=[
            "Origen",
            "Destino",
            "InformacionMutua"
        ]
    ).to_csv(
        "resultados/mst_kruskal.csv",
        index=False
    )

    # =========================
    # PRIM
    # =========================

    mst_p = nx.maximum_spanning_tree(
        G,
        algorithm="prim"
    )

    prim_data = []

    peso_prim = 0

    for u, v, d in mst_p.edges(data=True):

        peso_prim += d["weight"]

        prim_data.append([
            u,
            v,
            round(d["weight"], 6)
        ])

    pd.DataFrame(
        prim_data,
        columns=[
            "Origen",
            "Destino",
            "InformacionMutua"
        ]
    ).to_csv(
        "resultados/mst_prim.csv",
        index=False
    )

    # =========================
    # GRAFICO KRUSKAL
    # =========================

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(
        mst_k,
        seed=42
    )

    nx.draw(
        mst_k,
        pos,
        with_labels=True,
        node_size=2500,
        font_size=10
    )

    plt.title(
        "Maximum Spanning Tree - Kruskal"
    )

    plt.savefig(
        "static/img/mst_kruskal.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================
    # GRAFICO PRIM
    # =========================

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(
        mst_p,
        seed=42
    )

    nx.draw(
        mst_p,
        pos,
        with_labels=True,
        node_size=2500,
        font_size=10
    )

    plt.title(
        "Maximum Spanning Tree - Prim"
    )

    plt.savefig(
        "static/img/mst_prim.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {

        "variables": len(df.columns),

        "registros": len(df),

        "peso_prim": round(
            peso_prim,
            6
        ),

        "peso_kruskal": round(
            peso_kruskal,
            6
        )
    }