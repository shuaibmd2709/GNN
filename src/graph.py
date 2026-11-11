import numpy as np
import pandas as pd
from src.data import load_merged, add_entity_id

N_SNAPS, N_TRAIN, N_NODES = 15, 10, 5000
MIN_TX, MAX_TX = 3, 300      # DESIGN CHOICE: drop one-off cards and hub-like keys
POS_BOOST = 0.5              # >0 oversamples entities that ever had fraud
SEED = 42

def entity_table(df):
    g = df.groupby("entity_id")
    return pd.DataFrame({"n_tx": g.size(),
                         "n_snaps": g["snap"].nunique(),
                         "n_fraud": g["isFraud"].sum()})

def select_nodes(df, et):
    elig = et[(et.n_tx >= MIN_TX) & (et.n_tx <= MAX_TX) & (et.n_snaps >= 3)]
    w = 1.0 + POS_BOOST * (elig.n_fraud > 0).astype(float)
    ids = elig.sample(n=min(N_NODES, len(elig)), weights=w, random_state=SEED).index
    return elig, ids

if __name__ == "__main__":
    df = add_entity_id(load_merged())
    et = entity_table(df)
    print("top 5 entity sizes:", et.n_tx.nlargest(5).tolist())

    tr_e = df.loc[df.snap < N_TRAIN, "entity_id"].unique()
    te_e = df.loc[df.snap >= N_TRAIN, "entity_id"].unique()
    print("entities in test seen in train (per entity):", round(np.isin(te_e, tr_e).mean(), 3))

    elig, ids = select_nodes(df, et)
    print("eligible entities:", len(elig), "| selected nodes:", len(ids))

    sub = df[df.entity_id.isin(ids)]
    ns = sub.groupby(["entity_id", "snap"])["isFraud"].max().rename("y").reset_index()
    print("active (node, snapshot) pairs:", len(ns), "| positive rate:", round(ns.y.mean(), 4))
    print(ns.groupby("snap")["y"].agg(active="size", positives="sum", rate="mean").round(3))