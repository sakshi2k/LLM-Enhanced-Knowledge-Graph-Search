def load_graph(nodes_path: str, edges_path: str):
    import pandas as pd
    import networkx as nx

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    G = nx.DiGraph()

    # ---- ADD NODES (FIXED) ----
    for _, r in nodes.iterrows():
        attrs = r.to_dict()
        node_id = attrs.pop("id")   # remove id from attributes

        G.add_node(node_id, **attrs)

    # ---- ADD EDGES ----
    for _, r in edges.iterrows():
        attrs = r.to_dict()
        src = attrs.pop("source")
        tgt = attrs.pop("target")

        G.add_edge(src, tgt, **attrs)

    return G
