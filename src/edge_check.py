import numpy as np
import pandas as pd
from src.data import load_merged, add_entity_id
from src.graph import entity_table, select_nodes, N_SNAPS
from src.edges import build_edges, label_matrix

if __name__ == "__main__":
    df = add_entity_id(load_merged())
    _, ids = select_nodes(df, entity_table(df))
    node_of, edges = build_edges(df, ids)
    Y = label_matrix(df, node_of)
    base = (Y == 1).sum() / (Y >= 0).sum()

    rows = []
    for kind in ["email_addr", "browser_os_res", "device"]:
        src, dst, snp = [], [], []
        for s in range(N_SNAPS):
            for (i, j), ks in edges[s].items():
                if kind in ks:
                    src += [i, j]; dst += [j, i]; snp += [s, s]
        src, dst, snp = map(np.array, (src, dst, snp))
        fraud_src = Y[snp, src] == 1
        frac = (Y[snp, dst][fraud_src] == 1).mean()
        rows.append(dict(key=kind, edges=len(src) // 2,
                         edges_from_fraud=int(fraud_src.sum()),
                         fraud_nbr_frac=round(frac, 3),
                         lift=round(frac / base, 2)))
    print("base fraud rate:", round(base, 4))
    print(pd.DataFrame(rows).to_string(index=False))