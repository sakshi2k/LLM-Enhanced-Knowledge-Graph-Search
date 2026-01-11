import pandas as pd
import re

def normalize_topic_id(val):
    if not isinstance(val, str):
        return val

    # remove non-breaking spaces
    val = val.replace("\u00A0", "").strip()

    # match TOP<number>
    m = re.match(r"TOP\s*([0-9]+(\.[0-9]+)?)", val)
    if not m:
        return val

    num = int(float(m.group(1)))
    return f"TOP{num:06d}"


# ---- FIX NODES ----
nodes = pd.read_csv("./nodes.csv")

nodes["id"] = nodes["id"].apply(
    lambda x: normalize_topic_id(x) if str(x).startswith("TOP") else x
)

nodes.to_csv("nodes_fixed.csv", index=False)


# ---- FIX EDGES ----
edges = pd.read_csv("edges.csv")

edges["source"] = edges["source"].apply(
    lambda x: normalize_topic_id(x) if str(x).startswith("TOP") else x
)
edges["target"] = edges["target"].apply(
    lambda x: normalize_topic_id(x) if str(x).startswith("TOP") else x
)

edges.to_csv("edges_fixed.csv", index=False)

print("✔ CSVs fixed: nodes_fixed.csv, edges_fixed.csv")
