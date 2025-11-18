import json
import networkx as nx
import plotly.graph_objects as go

with open("outputs_sample/kg.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#edges or links
if "edges" in data:
    edges_key = "edges"
elif "links" in data:
    edges_key = "links"
else:
    raise KeyError(f"JSON must contain 'edges' or 'links'. Available keys: {list(data.keys())}")

# convert JSON → networkx graph
G = nx.node_link_graph(data, edges=edges_key)

print("Graph loaded:")
print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")


# node positions
pos = nx.spring_layout(G, k=0.3, iterations=50)


# Building edge 
edge_x = []
edge_y = []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode='lines',
    line=dict(width=1),
    hoverinfo='none'
)


# Building node 
node_x = []
node_y = []
texts = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

    # label
    attrs = G.nodes[node]
    label = attrs.get("name", attrs.get("label", node))
    typ = attrs.get("type", "")
    
    texts.append(f"{typ}: {label}")

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    marker=dict(
        size=8,
        color="blue",
        opacity=0.7
    ),
    text=texts,
    hoverinfo="text"
)


fig = go.Figure(data=[edge_trace, node_trace])

fig.update_layout(
    title="Knowledge Graph Visualization (Plotly)",
    title_x=0.5,
    showlegend=False,
    hovermode="closest",
    margin=dict(l=10, r=10, b=10, t=40),
)
fig.show()
