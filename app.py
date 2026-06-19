from flask import Flask
from flask import render_template

import pandas as pd

from algoritmo import analizar_dataset

app = Flask(__name__)


@app.route("/")
def inicio():

    datos = analizar_dataset(
        "d9_strong.csv"
    )

    entropias = pd.read_csv(
        "resultados/entropias.csv"
    )

    prim = pd.read_csv(
        "resultados/mst_prim.csv"
    )

    kruskal = pd.read_csv(
        "resultados/mst_kruskal.csv"
    )

    return render_template(
        "index.html",
        datos=datos,
        entropias=entropias.to_dict("records"),
        prim=prim.to_dict("records"),
        kruskal=kruskal.to_dict("records")
    )


if __name__ == "__main__":
    app.run(debug=True)