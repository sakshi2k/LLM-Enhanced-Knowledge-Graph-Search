import pandas as pd

# ---------- CONFIG ----------
NODES_IN = "nodes.csv"
EDGES_IN = "edges.csv"
NODES_OUT = "nodes_fixed.csv"
EDGES_OUT = "edges_fixed.csv"

# ---------- LOAD ----------
nodes = pd.read_csv(NODES_IN, dtype=str)
edges = pd.read_csv(EDGES_IN, dtype=str)

# ---------- BUILD ID MAP ----------
id_map = {}

for _, row in nodes.iterrows():
    old_id = row["id"]
    t = row["type"]

    if t == "Speaker" and old_id.startswith("SPE"):
        new_id = old_id.replace("SPE", "SPK", 1)
    elif t == "Speech" and old_id.startswith("SPE"):
        new_id = old_id.replace("SPE", "SPH", 1)
    else:
        new_id = old_id

    id_map[old_id] = new_id

nodes["id"] = nodes["id"].map(id_map)

# ---------- UPDATE EDGES IDS ----------
edges["source"] = edges["source"].map(id_map)
edges["target"] = edges["target"].map(id_map)

# ---------- NORMALIZE RELATIONSHIP TYPES ----------
def normalize_edge(row):
    t = row["type"].lower()

    if t == "delivered":
        return "DELIVERED"

    if t in ("covers", "contains"):
        if row["source"].startswith("SPK"):
            return "PROFILE_COVERS"
        if row["source"].startswith("SPH"):
            return "COVERS"

    if t == "held_on":
        return "HELD_ON"

    return row["type"].upper()


edges["type"] = edges.apply(normalize_edge, axis=1)

# ---------- SAVE ----------
nodes.to_csv(NODES_OUT, index=False)
edges.to_csv(EDGES_OUT, index=False)

print("✔ Migration complete")
print(f"✔ Nodes written to {NODES_OUT}")
print(f"✔ Edges written to {EDGES_OUT}")
