import itertools
import numpy as np
import pandas as pd
from src.data import load_merged, add_entity_id
from src.graph import entity_table, select_nodes, N_SNAPS

CAP = 20  # DESIGN CHOICE: skip key values shared by more than 20 nodes (too generic)

def link_keys(df):
    k = pd.DataFrame(index=df.index)
    k["device"] = df["DeviceInfo"].astype("string")
    k["browser_os_res"] = (df["id_31"].astype("string") + "|" +
                           df["id_30"].astype("string") + "|" +
                           df["id_33"].astype("string"))
    k["email_addr"] = (df["P_emaildomain"].astype("string") + "|" +
                       df["addr1"].astype("string"))
    return k

def build_edges(df, ids):
    node_of = {e: i for i, e in enumerate(ids)}
    sub = df[df.entity_id.isin(ids)]
    keys = link_keys(sub)
    sub = pd.concat([sub[["entity_id", "snap"]], keys], axis=1)
    edges = {s: {} for s in range(N_SNAPS)}      # snap -> {(i, j): {key types}}
    for kname in keys.columns:
        t = sub[["entity_id", "snap", kname]].dropna().drop_duplicates()
        for (s, v), g in t.groupby(["snap", kname]):
            nodes = sorted(node_of[e] for e in g.entity_id)
            if 2 <= len(nodes) <= CAP:
                for i, j in itertools.combinations(nodes, 2):
                    edges[s].setdefault((i, j), set()).add(kname)
    return node_of, edges

def label_matrix(df, node_of):
    sub = df[df.entity_id.isin(node_of)]
    ns = sub.groupby(["snap", "entity_id"])["isFraud"].max().reset_index()
    Y = np.full((N_SNAPS, len(node_of)), -1, dtype=int)   # -1 = node inactive
    Y[ns.snap.values, ns.entity_id.map(node_of).values] = ns.isFraud.values
    return Y

if __name__ == "__main__":
    df = add_entity_id(load_merged())
    et = entity_table(df)
    _, ids = select_nodes(df, et)
    node_of, edges = build_edges(df, ids)
    Y = label_matrix(df, node_of)
    N = len(node_of)

    rows = []
    for s in range(N_SNAPS):
        E = np.array(list(edges[s].keys()))
        active = Y[s] >= 0
        if len(E) == 0:
            rows.append(dict(snap=s, active=active.sum(), edges=0)); continue
        src = np.concatenate([E[:, 0], E[:, 1]])
        dst = np.concatenate([E[:, 1], E[:, 0]])
        deg = np.bincount(src, minlength=N)
        fraud_src = Y[s][src] == 1
        rows.append(dict(
            snap=s, active=active.sum(), edges=len(E),
            avg_deg=round(deg[active].mean(), 2),
            connected=round((deg[active] > 0).mean(), 3),
            fraud_nbr_frac=round((Y[s][dst][fraud_src] == 1).mean(), 3) if fraud_src.any() else np.nan,
        ))
    print(pd.DataFrame(rows).to_string(index=False))

    kinds = pd.Series([k for s in range(N_SNAPS) for ks in edges[s].values() for k in ks])
    print("\nedges by key type:\n", kinds.value_counts())