import os
import numpy as np
import pandas as pd
import torch
from src.data import load_merged, add_entity_id, chrono_split, preprocess
from src.graph import entity_table, select_nodes, N_SNAPS, N_TRAIN
from src.edges import build_edges, label_matrix

KEYS = ["email_addr", "browser_os_res", "device"]

if __name__ == "__main__":
    df = add_entity_id(load_merged())
    _, ids = select_nodes(df, entity_table(df))
    node_of, edges = build_edges(df, ids)
    Y = label_matrix(df, node_of)
    N = len(node_of)

    # preprocessing is fitted on train snapshots only, then applied to all
    train, test, feats = preprocess(*chrono_split(df, N_TRAIN))
    full = pd.concat([train, test])
    full = full[full.entity_id.isin(node_of)]

    # node feature at (snapshot, node) = mean of that entity's transactions in the window
    g = full.groupby(["snap", "entity_id"])
    Xm = g[feats].mean()
    Xm["log_ntx"] = np.log1p(g.size())
    X = np.zeros((N_SNAPS, N, len(feats) + 1), dtype=np.float32)
    s_idx = Xm.index.get_level_values("snap").values
    n_idx = Xm.index.get_level_values("entity_id").map(node_of).values
    X[s_idx, n_idx] = Xm.values.astype(np.float32)

    # edges: both directions, with a 3-way "which key linked them" attribute
    edge_index, edge_attr = [], []
    for s in range(N_SNAPS):
        items = list(edges[s].items())
        if items:
            ij = np.array([k for k, _ in items]).T
            ea = np.array([[kk in v for kk in KEYS] for _, v in items], dtype=np.float32)
            edge_index.append(torch.tensor(np.concatenate([ij, ij[::-1]], axis=1), dtype=torch.long))
            edge_attr.append(torch.tensor(np.concatenate([ea, ea])))
        else:
            edge_index.append(torch.zeros((2, 0), dtype=torch.long))
            edge_attr.append(torch.zeros((0, len(KEYS))))

    active = torch.tensor(Y >= 0)
    out = dict(X=torch.tensor(X), Y=torch.tensor(Y), active=active,
               edge_index=edge_index, edge_attr=edge_attr,
               entity_ids=[int(e) for e in ids], feats=feats + ["log_ntx"],
               edge_keys=KEYS, n_train=N_TRAIN)
    os.makedirs("data/processed", exist_ok=True)
    torch.save(out, "data/processed/graph.pt")

    xt = out["X"][:N_TRAIN][active[:N_TRAIN]]
    print("X shape [snaps, nodes, features]:", tuple(out["X"].shape))
    print("NaNs in X:", int(torch.isnan(out["X"]).sum()))
    print("train active rows:", xt.shape[0], "| feature mean/std overall:",
          round(xt.mean().item(), 3), round(xt.std().item(), 3))
    print("active nodes per snapshot:", active.sum(1).tolist())
    print("edge_index shapes:", [tuple(e.shape) for e in edge_index][:3], "...")
    print("saved:", round(os.path.getsize("data/processed/graph.pt") / 1e6, 1), "MB")