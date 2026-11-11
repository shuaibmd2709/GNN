import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW = "data/raw"

def load_merged():
    tx = pd.read_csv(f"{RAW}/train_transaction.csv")
    idt = pd.read_csv(f"{RAW}/train_identity.csv")
    df = tx.merge(idt, on="TransactionID", how="left")
    df["has_identity"] = df["DeviceType"].notna().astype(int)
    return df

def add_entity_id(df):
    key_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"]
    key = df[key_cols[0]].astype("string").fillna("NA")
    for c in key_cols[1:]:
        key = key + "_" + df[c].astype("string").fillna("NA")
    df["entity_id"] = pd.factorize(key)[0]
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    return df


DROP = ["TransactionID", "isFraud", "TransactionDT", "entity_id", "day"]

def chrono_split(df, train_frac=0.7):
    cutoff = df["TransactionDT"].quantile(train_frac)
    train = df[df["TransactionDT"] <= cutoff].copy()
    test = df[df["TransactionDT"] > cutoff].copy()
    return train, test

def preprocess(train, test, max_missing=0.5):
    feats = [c for c in train.columns if c not in DROP]

    # 1. drop columns that are mostly empty or constant (decided on train only)
    miss = train[feats].isna().mean()
    feats = [c for c in feats if miss[c] <= max_missing and train[c].nunique() > 1]

    num = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in feats if c not in num]

    # 2. categoricals -> integer codes learned on train; unseen -> 0
    for c in cat:
        cats = {v: i + 1 for i, v in enumerate(train[c].dropna().unique())}
        for d in (train, test):
            d[c] = d[c].map(cats).fillna(0).astype(int)

    # 3. numerics -> median fill, log amount, standardize (train stats only)
    med = train[num].median()
    for d in (train, test):
        d[num] = d[num].fillna(med)
        d["TransactionAmt"] = np.log1p(d["TransactionAmt"])
    scaler = StandardScaler().fit(train[num])
    for d in (train, test):
        d[num] = scaler.transform(d[num])

    return train, test, feats

if __name__ == "__main__":
    print("loading...")
    df = add_entity_id(load_merged())
    train, test = chrono_split(df)
    print("train rows:", len(train), "| days", train["day"].min(), "->", train["day"].max())
    print("test rows: ", len(test), "| days", test["day"].min(), "->", test["day"].max())
    print("fraud rate train/test:", round(train["isFraud"].mean(), 4), round(test["isFraud"].mean(), 4))
    print("entities in test also seen in train:",
          round(test["entity_id"].isin(train["entity_id"]).mean(), 3))

    train, test, feats = preprocess(train, test)
    print("features kept:", len(feats))
    print("NaNs left (train/test):", int(train[feats].isna().sum().sum()), int(test[feats].isna().sum().sum()))
    print("train mean/std of 3 features:", train[feats[:3]].mean().round(3).tolist(), train[feats[:3]].std().round(3).tolist())