import os
import re
import pandas as pd
import spacy
import numpy as np
import json
import unicodedata
from tqdm import tqdm
import networkx as nx
import argparse


def sanitize_id(value: str) -> str:
    if not value:
        return "unknown"
    try:
        s = unicodedata.normalize("NFKD", value)
        s = s.encode("ascii", errors="ignore").decode("ascii")
    except Exception:
        s = value
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    if not s:
        return "unknown"
    return s


CANONICAL_MAP = {
    # sanitized form -> canonical node id
    "ecb": "org:european_central_bank",
    "european_central_bank": "org:european_central_bank"
}

def canonicalize_minimal(prefix: str, raw_name: str) -> str:
    sid = sanitize_id(raw_name)
    if sid in CANONICAL_MAP:
        return CANONICAL_MAP[sid]
    return f"{prefix}:{sid}"


def light_clean_text(text: str) -> str:
    if not text:
        return ""
    # removing "SPEECH" header and collapse whitespace
    t = re.sub(r'^\s*SPEECH\s*', '', text, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def build_graph(df: pd.DataFrame, nlp, log_every: int = 0, limit: int = 0) -> nx.MultiDiGraph:

    G = nx.MultiDiGraph()
    total_rows = len(df) if (limit is None or limit <= 0) else min(limit, len(df))

    texts = [
        " ".join([row.get("title","") or "", row.get("subtitle","") or "", light_clean_text(row.get("contents","") or "")])
        for _, row in df.iterrows()
    ]

    iterator = zip(df.iterrows(), nlp.pipe(texts, batch_size=50))
    processed = 0
    for (df_index, row), doc in tqdm(iterator, total=len(texts), desc="Building KG"):
        if limit and limit > 0 and processed >= limit:
            break
        processed += 1


        date_raw = (row.get("date") or "").strip()
        speakers_raw = (row.get("speakers") or "").strip()
        title_raw = (row.get("title") or "").strip()
        subtitle_raw = (row.get("subtitle") or "").strip()

        doc_id = f"doc:{df_index}"
        speech_id = f"speech:{df_index}"

        G.add_node(doc_id, type="Document", title=title_raw, date=date_raw, label=title_raw or doc_id)
        G.add_node(speech_id, type="Speech", title=title_raw, date=date_raw, label=title_raw or speech_id)
        G.add_edge(doc_id, speech_id, relation="same_as", source_row=df_index)

        # date node
        if date_raw:
            date_node = f"date:{sanitize_id(date_raw)}"
            if date_node not in G:
                G.add_node(date_node, type="Date", value=date_raw, label=date_raw)
            G.add_edge(speech_id, date_node, relation="held_on", source_row=df_index)

        # speakers -> person nodes
        speaker_nodes = []
        if speakers_raw:
            # assume comma-separated list of speakers
            for sp in [s.strip() for s in speakers_raw.split(",") if s.strip()]:
                pid = canonicalize_minimal("person", sp)
                if pid not in G:
                    # keep original name in 'name' attr
                    G.add_node(pid, type="Person", name=sp, label=sp)
                G.add_edge(pid, speech_id, relation="delivered", source_row=df_index)
                speaker_nodes.append(pid)

        # NER entities 
        for s_idx, sent in enumerate(doc.sents):
            for ent in sent.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE"):
                    ent_text = ent.text.strip()
                    if not ent_text:
                        continue
                    prefix_map = {"PERSON":"person", "ORG":"org", "GPE":"loc", "LOC":"loc", "DATE":"date"}
                    prefix = prefix_map.get(ent.label_, "entity")
                    node_id = canonicalize_minimal(prefix, ent_text)
                    if node_id not in G:
                        G.add_node(node_id, type=ent.label_, name=ent_text, label=ent_text)
                    # mention edge
                    G.add_edge(doc_id, node_id, relation="mentions", source_row=df_index, sentence_index=s_idx)

        topics = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # requires at least 2 words to avoid tiny words
            if len(chunk_text.split()) >= 2:
                t = re.sub(r'^(the|a|an)\s+', '', chunk_text, flags=re.IGNORECASE)
                topics.append(t)
        topics = list(dict.fromkeys(topics))[:5]  
        for t in topics:
            tid = f"topic:{sanitize_id(t)}"
            if tid not in G:
                G.add_node(tid, type="Topic", name=t, label=t)
            for spn in speaker_nodes:
                G.add_edge(spn, tid, relation="covers", source_row=df_index)
            G.add_edge(speech_id, tid, relation="discusses", source_row=df_index)

        # logging

        if log_every and processed % log_every == 0:
            print(f"[build] processed {processed}/{total_rows}")

    return G


def export_graph(G: nx.MultiDiGraph, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    exported = {}
    # JSON (nodes+edges)
    nodes = []
    for nid, attrs in G.nodes(data=True):
        row = {"id": nid}
        row.update(attrs)
        nodes.append(row)
    edges = []
    for s, t, attrs in G.edges(data=True):
        e = {"source": s, "target": t}
        e.update(attrs)
        edges.append(e)
    json_path = os.path.join(out_dir, "kg.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
    exported["json"] = json_path

    # Neo4j CSV style
    nodes_rows = []
    for n, attrs in G.nodes(data=True):
        nodes_rows.append({
            "id": n,
            "label": attrs.get("type", ""),
            "name": attrs.get("name", attrs.get("label", "")),
            "title": attrs.get("title", ""),
            "date": attrs.get("date", "")
        })
    rels_rows = []
    for s, t, attrs in G.edges(data=True):
        rels_rows.append({
            "start_id": s,
            "end_id": t,
            "type": attrs.get("relation", "RELATED_TO")
        })
    nodes_csv = os.path.join(out_dir, "kg_nodes.csv")
    rels_csv = os.path.join(out_dir, "kg_rels.csv")
    pd.DataFrame(nodes_rows).to_csv(nodes_csv, index=False)
    pd.DataFrame(rels_rows).to_csv(rels_csv, index=False)
    exported["nodes_csv"] = nodes_csv
    exported["rels_csv"] = rels_csv

    return exported


def main():
    #nlp = spacy.load("en_core_web_sm")
    #print("Loaded spaCy model:", nlp.pipe_names)

    #csv_path = "sample_speeches.csv"
    #df = pd.read_csv(csv_path, sep="|", dtype=str, keep_default_na=False)
    # sample = df.loc[0, 'contents']

    parser = argparse.ArgumentParser(description="Build a small KG from ECB-style CSV.")
    parser.add_argument("--csv", type=str, default="sample_speeches.csv", help="Path to CSV (pipe-separated)")
    #parser.add_argument("--csv", type=str, default="all_ECB_speeches.csv", help="Path to CSV (pipe-separated)")
    #parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--out", type=str, default="outputs_sample", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--log-every", type=int, default=0, help="Log progress every N rows (0=disabled)")
    args = parser.parse_args()

    # read csv
    df = pd.read_csv(args.csv, sep="|", dtype=str, keep_default_na=False)
    print("Loaded rows:", len(df))

    # load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # build graph
    print("Building graph...")
    G = build_graph(df, nlp, log_every=args.log_every, limit=args.limit)
    print("Built graph:", G.number_of_nodes(), "nodes;", G.number_of_edges(), "edges")

    # export
    print("Exporting...")
    files = export_graph(G, args.out)
    for k, p in files.items():
        print("Export:", k, p)
    print("Done.")
    
    print("Degree of Europeancentral bank: ", G.degree("org:european_central_bank"))
    print(list(G.edges("org:european_central_bank", data=True)))
    # or for directed graphs:
    print(list(G.in_edges("org:european_central_bank", data=True)))
    print(list(G.out_edges("org:european_central_bank", data=True)))
    print(list(G.in_edges("org:european_central_bank", keys=True, data=True)))
    unique_edges = {(u,v) for u,v,_ in G.in_edges("org:european_central_bank", keys=True)}
    print("\n\nUnique edges :", unique_edges)

# yields tuples: (u, v, key, data)

'''
    names = ["", "ECB", "European Central Bank", "Pádraig Ó hUiginn", "European's people"]
    for n in names:
        print(n, "->", santize_ID(n))

    print("\nEntities (text, label):")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    print('\nNoun Chunks: (first 20):')
    for i, chunk in enumerate(doc.noun_chunks):
        if i >= 20:
            break
        print("-", chunk.text)


    #print("Before(start):", sample[:120])
    #print("After (cleaned):", light_clean_text(sample)[:120])

    # Now, printing the head will show the cleaned contents

    df['contents'] = df['contents'].apply(light_clean_text) 
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print(df.head().to_string())
'''

if __name__ == "__main__":
    main()