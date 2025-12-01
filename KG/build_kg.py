"""
Knowledge graph construction pipeline for ECB speeches.

This script reads the provided CSV, cleans and normalizes the text (including
abbreviation expansion), extracts entities and relations with spaCy and
Sentence-BERT, builds a NetworkX graph following the schema
Speaker --delivered--> Speech --held_on--> Date and Speaker --covers--> Topic,
evaluates the resulting graph, and exports Neo4j-ready CSV files alongside a
JSON metrics report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from dateutil import parser as date_parser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from spacy.util import filter_spans
from tqdm import tqdm

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "spaCy is required. Install dependencies inside the provided virtualenv."
    ) from exc


DEFAULT_ABBREVIATIONS = {
    "ecb": "European Central Bank",
    "vol.": "volume",
    "vol": "volume",
    "us": "United States",
    "u.s.": "United States",
    "u.k.": "United Kingdom",
    "uk": "United Kingdom",
    "eu": "European Union",
    "eurozone": "Euro Area",
    "fed": "Federal Reserve",
    "gdp": "gross domestic product",
    "fx": "foreign exchange",
    "cb": "central bank",
    "cbanks": "central banks",
    "imo": "International Maritime Organization",
    "eba": "European Banking Authority",
    "srep": "Supervisory Review and Evaluation Process",
    "dlt": "distributed ledger technology",
}

NODE_TYPES = ("Speaker", "Speech", "Topic", "Date")
REL_DELIVERED = "delivered"
REL_COVERS = "covers"
REL_HELD_ON = "held_on"

TOPIC_LABELS = {"ORG", "EVENT", "LAW", "WORK_OF_ART", "PRODUCT", "NORP"}
STOP_PATTERN = re.compile(r"^(and|or|but|however|therefore)$", re.IGNORECASE)


def read_abbreviation_config(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Abbreviation config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    normalized = {}
    for key, value in data.items():
        if not key or not value:
            continue
        normalized[key.strip().lower()] = value.strip()
    return normalized


class AbbreviationExpander:
    def __init__(self, mapping: Dict[str, str]):
        base = {k.lower(): v for k, v in mapping.items()}
        self.mapping = base
        if base:
            pattern = "|".join(re.escape(k) for k in sorted(base, key=len, reverse=True))
            self.regex = re.compile(rf"\b({pattern})\b", flags=re.IGNORECASE)
        else:
            self.regex = None
        self.stats = Counter()

    def expand(self, text: str) -> str:
        if not text or not self.regex:
            return text

        def repl(match: re.Match) -> str:
            original = match.group(0)
            replacement = self.mapping.get(original.lower(), original)
            self.stats[original.lower()] += 1
            if original.isupper():
                return replacement.upper()
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        return self.regex.sub(repl, text)


class TextCleaner:
    def __init__(self, expander: AbbreviationExpander):
        self.expander = expander
        self.whitespace_re = re.compile(r"\s+")
        self.speech_marker = re.compile(r"(?i)^(speech)(\s+|:)+")

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        text = self.expander.expand(text)
        text = self.whitespace_re.sub(" ", text)
        text = self.speech_marker.sub("", text, count=1)
        return text.strip()


def safe_parse_date(value: str) -> Tuple[Optional[str], bool]:
    if not isinstance(value, str) or not value.strip():
        return None, False
    try:
        dt = date_parser.parse(value, dayfirst=False, yearfirst=True)
        return dt.date().isoformat(), True
    except (ValueError, TypeError, OverflowError):
        return None, False


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", label or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


@dataclass
class TopicCandidate:
    label: str
    score: float
    embedding: np.ndarray


class TopicExtractor:
    def __init__(
        self,
        nlp: Language,
        embedder: SentenceTransformer,
        max_chars: int = 8000,
        max_topics: int = 3,
        cluster_threshold: float = 0.72,
    ):
        self.nlp = nlp
        self.embedder = embedder
        self.max_chars = max_chars
        self.max_topics = max_topics
        self.cluster_threshold = cluster_threshold

    def _collect_phrases(self, doc) -> List[str]:
        phrases: List[str] = []
        for ent in filter_spans(doc.ents):
            if ent.label_ in TOPIC_LABELS:
                normalized = normalize_label(ent.text)
                if normalized and not STOP_PATTERN.match(normalized.lower()):
                    phrases.append(normalized)
        for chunk in doc.noun_chunks:
            candidate = normalize_label(chunk.text)
            if len(candidate.split()) <= 1:
                continue
            if candidate and not STOP_PATTERN.match(candidate.lower()):
                phrases.append(candidate)
        for sent in doc.sents:
            candidate = normalize_label(sent.text)
            if len(candidate.split()) < 4:
                continue
            phrases.append(candidate)
        return phrases

    def extract(self, text: str) -> List[TopicCandidate]:
        truncated = (text or "")[: self.max_chars]
        if not truncated.strip():
            return []
        doc = self.nlp(truncated)
        phrases = list(dict.fromkeys(self._collect_phrases(doc)))
        if not phrases:
            return []
        embeddings = self.embedder.encode(
            phrases,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        clusters: List[Dict[str, object]] = []
        for phrase, vector in zip(phrases, embeddings):
            assigned = False
            for cluster in clusters:
                centroid = cluster["centroid"]  # type: ignore[assignment]
                score = float(np.dot(vector, centroid))
                if score >= self.cluster_threshold:
                    cluster["members"].append((phrase, vector))
                    cluster["centroid"] = self._normalized_centroid(
                        cluster["members"]
                    )
                    assigned = True
                    break
            if not assigned:
                clusters.append(
                    {
                        "members": [(phrase, vector)],
                        "centroid": vector,
                    }
                )
        scored_topics: List[TopicCandidate] = []
        for cluster in clusters:
            members: List[Tuple[str, np.ndarray]] = cluster["members"]
            centroid = cluster["centroid"]
            member_scores = [
                float(np.dot(vec, centroid)) for _, vec in members
            ]
            avg_score = float(np.mean(member_scores))
            canonical = self._cluster_label(members)
            scored_topics.append(
                TopicCandidate(label=canonical, score=avg_score, embedding=centroid)
            )
        scored_topics.sort(key=lambda c: (c.score, len(c.label)), reverse=True)
        return scored_topics[: self.max_topics]

    @staticmethod
    def _normalized_centroid(
        members: List[Tuple[str, np.ndarray]]
    ) -> np.ndarray:
        matrix = np.vstack([vec for _, vec in members])
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return centroid
        return centroid / norm

    @staticmethod
    def _cluster_label(members: List[Tuple[str, np.ndarray]]) -> str:
        unique = []
        seen = set()
        for phrase, _ in sorted(members, key=lambda pair: (-len(pair[0]), pair[0])):
            if phrase in seen:
                continue
            seen.add(phrase)
            unique.append(phrase)
            if len(unique) == 2:
                break
        if not unique:
            return members[0][0]
        if len(unique) == 1:
            return unique[0]
        return " | ".join(unique)


class KnowledgeGraphBuilder:
    def __init__(
        self,
        embedder: SentenceTransformer,
        topic_threshold: float = 0.78,
    ):
        self.graph = nx.MultiDiGraph()
        self.node_index: Dict[Tuple[str, str], str] = {}
        self.embedder = embedder
        self.topic_threshold = topic_threshold
        self.topic_registry_vectors: List[np.ndarray] = []
        self.topic_registry_ids: List[str] = []
        self.topic_aliases: Dict[str, set] = {}
        self.topic_counts: Dict[str, int] = {}
        self.date_stats = Counter()
        self.topic_metric_vectors: List[np.ndarray] = []
        self.topic_metric_ids: List[str] = []

    def _next_id(self, prefix: str) -> str:
        count = sum(1 for node in self.graph.nodes if str(node).startswith(prefix))
        return f"{prefix}{count+1:06d}"

    def _get_or_create_node(
        self,
        label: str,
        node_type: str,
        **attrs,
    ) -> str:
        key = (node_type, label.lower())
        if key in self.node_index:
            node_id = self.node_index[key]
            self.graph.nodes[node_id].update(attrs)
            return node_id
        node_id = self._next_id(node_type[:3].upper())
        node_attrs = {"type": node_type, "label": label}
        node_attrs.update(attrs)
        self.graph.add_node(node_id, **node_attrs)
        self.node_index[key] = node_id
        return node_id

    def upsert_topic(self, topic: TopicCandidate) -> str:
        vector = topic.embedding
        if not self.topic_registry_vectors:
            topic_id = self._get_or_create_node(
                topic.label,
                "Topic",
                confidence=topic.score,
                aliases=[topic.label],
                mentions=1,
            )
            self.topic_registry_vectors.append(vector)
            self.topic_registry_ids.append(topic_id)
            self.topic_aliases[topic_id] = {topic.label}
            self.topic_counts[topic_id] = 1
            return topic_id
        matrix = np.vstack(self.topic_registry_vectors)
        sims = cosine_similarity([vector], matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= self.topic_threshold:
            topic_id = self.topic_registry_ids[best_idx]
            self.topic_counts[topic_id] += 1
            self.topic_aliases[topic_id].add(topic.label)
            self.graph.nodes[topic_id]["aliases"] = sorted(
                self.topic_aliases[topic_id]
            )
            self.graph.nodes[topic_id]["mentions"] = self.topic_counts[topic_id]
            self.graph.nodes[topic_id]["confidence"] = (
                self.graph.nodes[topic_id].get("confidence", topic.score) + topic.score
            ) / 2
            self.topic_registry_vectors[best_idx] = self._normalize_vector(
                (
                    self.topic_registry_vectors[best_idx]
                    * (self.topic_counts[topic_id] - 1)
                )
                + vector
            )
            return topic_id
        topic_id = self._get_or_create_node(
            topic.label,
            "Topic",
            confidence=topic.score,
            aliases=[topic.label],
            mentions=1,
        )
        self.topic_registry_vectors.append(vector)
        self.topic_registry_ids.append(topic_id)
        self.topic_aliases[topic_id] = {topic.label}
        self.topic_counts[topic_id] = 1
        return topic_id

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add_speech(
        self,
        row_id: int,
        speaker_label: str,
        speech_title: str,
        speech_subtitle: str,
        speech_text: str,
        topics: Sequence[TopicCandidate],
        date_iso: Optional[str],
    ):
        speaker_id = self._get_or_create_node(
            speaker_label,
            "Speaker",
            source="csv",
        )
        speech_id = self._get_or_create_node(
            f"{row_id}:{speech_title}",
            "Speech",
            title=speech_title,
            subtitle=speech_subtitle,
            text_excerpt=speech_text[:400],
            source_row=row_id,
        )
        self.graph.add_edge(
            speaker_id,
            speech_id,
            key=f"{speech_id}-delivered",
            type=REL_DELIVERED,
            confidence=1.0,
            source_row=row_id,
        )
        topic_ids: List[str] = []
        for topic in topics:
            topic_id = self.upsert_topic(topic)
            topic_ids.append(topic_id)
            self.graph.add_edge(
                speaker_id,
                topic_id,
                key=f"{speech_id}-{topic_id}-covers",
                type=REL_COVERS,
                confidence=topic.score,
                source_row=row_id,
            )
        if date_iso:
            date_id = self._get_or_create_node(
                date_iso,
                "Date",
                iso=date_iso,
            )
            self.graph.add_edge(
                speech_id,
                date_id,
                key=f"{speech_id}-held_on",
                type=REL_HELD_ON,
                confidence=1.0,
                source_row=row_id,
            )
        self.topic_metric_vectors.extend([t.embedding for t in topics])
        self.topic_metric_ids.extend(topic_ids)

    def compute_metrics(self) -> Dict[str, object]:
        graph = self.graph
        undirected = graph.to_undirected()
        # Convert multigraph to simple graph for clustering calculation
        simple_graph = nx.Graph(undirected)
        
        # Basic graph statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        components = nx.number_connected_components(undirected)
        
        # Clustering coefficient
        clustering = (
            nx.average_clustering(simple_graph) if simple_graph.number_of_nodes() else 0.0
        )
        
        # Degree statistics
        degrees = dict(graph.degree())
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        avg_degree = np.mean(list(degrees.values())) if degrees else 0.0
        avg_in_degree = np.mean(list(in_degrees.values())) if in_degrees else 0.0
        avg_out_degree = np.mean(list(out_degrees.values())) if out_degrees else 0.0
        
        # Graph density
        max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 0
        density = (2 * num_edges / max_edges) if max_edges > 0 else 0.0
        
        # Largest component analysis
        if components > 0:
            largest_cc = max(nx.connected_components(undirected), key=len)
            largest_cc_graph = simple_graph.subgraph(largest_cc)
            largest_cc_size = len(largest_cc)
            try:
                diameter = nx.diameter(largest_cc_graph) if largest_cc_size > 1 else 0
                radius = nx.radius(largest_cc_graph) if largest_cc_size > 1 else 0
            except (nx.NetworkXError, nx.NetworkXPointlessConcept):
                diameter = 0
                radius = 0
        else:
            largest_cc_size = 0
            diameter = 0
            radius = 0
        
        # Node and edge type distributions
        node_counts = Counter(nx.get_node_attributes(graph, "type").values())
        edge_counts = Counter(
            data["type"] for _, _, data in graph.edges(data=True)
        )
        
        # Entity resolution quality (unique entities vs total mentions)
        entity_resolution = self._entity_resolution_metrics()
        
        # Topic metrics
        topic_metrics = self._topic_metrics()
        
        # Precision/Recall/F1 metrics
        prf_metrics = self._compute_precision_recall_f1()
        
        return {
            "graph": {
                "nodes": num_nodes,
                "edges": num_edges,
                "density": round(density, 6),
                "components": components,
                "largest_component_size": largest_cc_size,
                "largest_component_ratio": round(largest_cc_size / num_nodes, 4) if num_nodes > 0 else 0.0,
                "diameter": diameter,
                "radius": radius,
                "average_clustering": round(clustering, 6),
                "average_degree": round(avg_degree, 2),
                "average_in_degree": round(avg_in_degree, 2),
                "average_out_degree": round(avg_out_degree, 2),
                "nodes_by_type": dict(node_counts),
                "edges_by_type": dict(edge_counts),
            },
            "precision_recall_f1": prf_metrics,
            "entity_resolution": entity_resolution,
            "topics": topic_metrics,
            "dates": self.date_stats,
        }

    def _compute_precision_recall_f1(self) -> Dict[str, object]:
        """
        Compute Precision, Recall, and F1 scores based on schema compliance and completeness.
        Without ground truth, we evaluate:
        - Schema compliance: expected entity types and relations
        - Completeness: required fields present
        - Coverage: extraction from input data
        """
        graph = self.graph
        
        # Expected schema: Speaker, Speech, Topic, Date
        expected_node_types = {"Speaker", "Speech", "Topic", "Date"}
        expected_edge_types = {"delivered", "covers", "held_on"}
        
        # Actual extracted types
        actual_node_types = set(nx.get_node_attributes(graph, "type").values())
        actual_edge_types = set(
            data["type"] for _, _, data in graph.edges(data=True)
        )
        
        # Schema compliance metrics
        # Note: These measure if expected types exist, not extraction quality
        node_type_precision = len(actual_node_types & expected_node_types) / len(actual_node_types) if actual_node_types else 0.0
        node_type_recall = len(actual_node_types & expected_node_types) / len(expected_node_types) if expected_node_types else 0.0
        node_type_f1 = (2 * node_type_precision * node_type_recall / (node_type_precision + node_type_recall)) if (node_type_precision + node_type_recall) > 0 else 0.0
        
        edge_type_precision = len(actual_edge_types & expected_edge_types) / len(actual_edge_types) if actual_edge_types else 0.0
        edge_type_recall = len(actual_edge_types & expected_edge_types) / len(expected_edge_types) if expected_edge_types else 0.0
        edge_type_f1 = (2 * edge_type_precision * edge_type_recall / (edge_type_precision + edge_type_recall)) if (edge_type_precision + edge_type_recall) > 0 else 0.0
        
        # Additional quality metrics (without ground truth)
        # These provide more insight into graph quality
        node_type_coverage = {
            node_type: len([n for n, d in graph.nodes(data=True) if d.get("type") == node_type])
            for node_type in expected_node_types
        }
        edge_type_coverage = {
            edge_type: len([(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d.get("type") == edge_type])
            for edge_type in expected_edge_types
        }
        
        # Completeness: Check if required relations exist
        # Expected: Each Speech should have: delivered (from Speaker), held_on (to Date), and topics (covers from Speaker)
        speech_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("type") == "Speech"
        ]
        
        total_speeches = len(speech_nodes)
        speeches_with_speaker = 0
        speeches_with_date = 0
        speeches_with_topic = 0
        complete_speeches = 0
        
        for speech_id in speech_nodes:
            has_speaker = any(
                graph[u][v][k].get("type") == "delivered" and v == speech_id
                for u, v, k in graph.in_edges(speech_id, keys=True)
            )
            has_date = any(
                graph[u][v][k].get("type") == "held_on" and u == speech_id
                for u, v, k in graph.out_edges(speech_id, keys=True)
            )
            # Check if speaker of this speech has topic connections
            speaker_has_topic = False
            speaker_id = None
            for u, v, k in graph.in_edges(speech_id, keys=True):
                if graph[u][v][k].get("type") == "delivered":
                    speaker_id = u
                    break
            if speaker_id:
                speaker_has_topic = any(
                    graph[u][v][k].get("type") == "covers"
                    for u, v, k in graph.out_edges(speaker_id, keys=True)
                    if graph.nodes[v].get("type") == "Topic"
                )
            
            if has_speaker:
                speeches_with_speaker += 1
            if has_date:
                speeches_with_date += 1
            if speaker_has_topic:
                speeches_with_topic += 1
            if has_speaker and has_date and speaker_has_topic:
                complete_speeches += 1
        
        # Completeness metrics
        speaker_completeness = speeches_with_speaker / total_speeches if total_speeches > 0 else 0.0
        date_completeness = speeches_with_date / total_speeches if total_speeches > 0 else 0.0
        topic_completeness = speeches_with_topic / total_speeches if total_speeches > 0 else 0.0
        overall_completeness = complete_speeches / total_speeches if total_speeches > 0 else 0.0
        
        # Date parsing accuracy (we have this from date_stats)
        date_success = self.date_stats.get("success", 0)
        date_total = date_success + self.date_stats.get("failure", 0)
        date_accuracy = date_success / date_total if date_total > 0 else 0.0
        
        return {
            "schema_compliance": {
                "node_types": {
                    "precision": round(node_type_precision, 4),
                    "recall": round(node_type_recall, 4),
                    "f1": round(node_type_f1, 4),
                    "note": "Measures if expected node types exist in graph (schema compliance, not extraction quality)",
                },
                "edge_types": {
                    "precision": round(edge_type_precision, 4),
                    "recall": round(edge_type_recall, 4),
                    "f1": round(edge_type_f1, 4),
                    "note": "Measures if expected edge types exist in graph (schema compliance, not extraction quality)",
                },
                "coverage": {
                    "node_types": node_type_coverage,
                    "edge_types": edge_type_coverage,
                },
            },
            "completeness": {
                "speaker_relation": round(speaker_completeness, 4),
                "date_relation": round(date_completeness, 4),
                "topic_relation": round(topic_completeness, 4),
                "overall": round(overall_completeness, 4),
            },
            "accuracy": {
                "date_parsing": round(date_accuracy, 4),
            },
            "macro_f1": round((node_type_f1 + edge_type_f1 + overall_completeness + date_accuracy) / 4, 4),
        }
    
    def _entity_resolution_metrics(self) -> Dict[str, object]:
        """Track entity resolution quality - how well we merge aliases."""
        speaker_nodes = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get("type") == "Speaker"
        ]
        topic_nodes = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get("type") == "Topic"
        ]
        
        # Count unique canonical names vs total mentions
        speaker_canonicals = len(set(data.get("label", "") for _, data in speaker_nodes))
        topic_canonicals = len(set(data.get("label", "") for _, data in topic_nodes))
        
        # Count aliases (if stored)
        speaker_aliases = sum(
            len(data.get("aliases", [])) for _, data in speaker_nodes
        )
        topic_aliases = sum(
            len(data.get("aliases", [])) for _, data in topic_nodes
        )
        
        return {
            "speakers": {
                "unique_entities": speaker_canonicals,
                "total_nodes": len(speaker_nodes),
                "aliases_resolved": speaker_aliases,
                "resolution_ratio": round(speaker_canonicals / len(speaker_nodes), 4) if speaker_nodes else 0.0,
            },
            "topics": {
                "unique_entities": topic_canonicals,
                "total_nodes": len(topic_nodes),
                "aliases_resolved": topic_aliases,
                "resolution_ratio": round(topic_canonicals / len(topic_nodes), 4) if topic_nodes else 0.0,
            },
        }
    
    def _topic_metrics(self) -> Dict[str, float]:
        if not self.topic_metric_vectors:
            return {"avg_intra_similarity": 0.0, "avg_inter_similarity": 0.0}
        matrix = np.vstack(self.topic_metric_vectors)
        if matrix.shape[0] < 2:
            return {"avg_intra_similarity": 0.0, "avg_inter_similarity": 0.0}
        sims = cosine_similarity(matrix)
        intra_scores = []
        inter_scores = []
        for i in range(len(self.topic_metric_ids)):
            for j in range(i + 1, len(self.topic_metric_ids)):
                if self.topic_metric_ids[i] == self.topic_metric_ids[j]:
                    intra_scores.append(sims[i, j])
                else:
                    inter_scores.append(sims[i, j])
        avg_intra = float(np.mean(intra_scores)) if intra_scores else 0.0
        avg_inter = float(np.mean(inter_scores)) if inter_scores else 0.0
        return {
            "avg_intra_similarity": round(avg_intra, 4),
            "avg_inter_similarity": round(avg_inter, 4),
            "topic_nodes": len({nid for nid in self.topic_metric_ids}),
            "coherence_ratio": round(avg_intra / avg_inter, 4) if avg_inter > 0 else 0.0,
        }

    def export(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        nodes_path = os.path.join(output_dir, "nodes.csv")
        edges_path = os.path.join(output_dir, "edges.csv")
        pd.DataFrame(
            [
                {"id": node_id, **data}
                for node_id, data in self.graph.nodes(data=True)
            ]
        ).to_csv(nodes_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        pd.DataFrame(
            [
                {
                    "source": u,
                    "target": v,
                    "key": k,
                    **data,
                }
                for u, v, k, data in self.graph.edges(data=True, keys=True)
            ]
        ).to_csv(edges_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def load_dataframe(path: str, delimiter: str, sample_size: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        on_bad_lines="skip",
    )
    if sample_size:
        return df.head(sample_size)
    return df


def build_pipeline(args: argparse.Namespace):
    abbr_map = DEFAULT_ABBREVIATIONS.copy()
    abbr_map.update(read_abbreviation_config(args.abbreviation_config))
    expander = AbbreviationExpander(abbr_map)
    cleaner = TextCleaner(expander)

    nlp = spacy.load("en_core_web_lg")
    nlp.max_length = 3_000_000
    embedder = SentenceTransformer(args.embedding_model)
    topic_extractor = TopicExtractor(
        nlp,
        embedder,
        max_topics=args.max_topics,
        cluster_threshold=args.topic_cluster_threshold,
    )
    kg_builder = KnowledgeGraphBuilder(
        embedder,
        topic_threshold=args.topic_merge_threshold,
    )
    df = load_dataframe(args.input, args.delimiter, args.sample_size)
    tqdm.pandas(desc="Processing speeches")
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Rows",
    ):
        speaker = cleaner.clean(row.get("speakers", ""))
        title = cleaner.clean(row.get("title", ""))
        subtitle = cleaner.clean(row.get("subtitle", ""))
        contents = cleaner.clean(row.get("contents", ""))
        date_iso, ok = safe_parse_date(row.get("date", ""))
        kg_builder.date_stats["success" if ok else "failure"] += 1
        topics = topic_extractor.extract(" ".join([title, subtitle, contents]))
        if not speaker or not title:
            continue
        kg_builder.add_speech(
            row_id=idx,
            speaker_label=speaker,
            speech_title=title,
            speech_subtitle=subtitle,
            speech_text=contents,
            topics=topics,
            date_iso=date_iso,
        )
    kg_builder.export(args.output_dir)
    metrics = kg_builder.compute_metrics()
    metrics["abbreviations"] = expander.stats
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    
    # Display formatted metrics
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH EVALUATION METRICS")
    print("="*80)
    print(f"\nExported nodes and edges to: {args.output_dir}")
    
    # Graph structure metrics
    graph_metrics = metrics["graph"]
    print("\n📊 GRAPH STRUCTURE METRICS:")
    print(f"  • Total Nodes: {graph_metrics['nodes']:,}")
    print(f"  • Total Edges: {graph_metrics['edges']:,}")
    print(f"  • Graph Density: {graph_metrics['density']:.6f}")
    print(f"  • Connected Components: {graph_metrics['components']}")
    print(f"  • Largest Component: {graph_metrics['largest_component_size']:,} nodes ({graph_metrics['largest_component_ratio']*100:.1f}%)")
    print(f"  • Diameter: {graph_metrics['diameter']}")
    print(f"  • Radius: {graph_metrics['radius']}")
    print(f"  • Average Clustering Coefficient: {graph_metrics['average_clustering']:.6f}")
    print(f"  • Average Degree: {graph_metrics['average_degree']:.2f}")
    print(f"    - Average In-Degree: {graph_metrics['average_in_degree']:.2f}")
    print(f"    - Average Out-Degree: {graph_metrics['average_out_degree']:.2f}")
    
    print("\n📦 NODES BY TYPE:")
    for node_type, count in sorted(graph_metrics["nodes_by_type"].items()):
        print(f"  • {node_type}: {count:,}")
    
    print("\n🔗 EDGES BY TYPE:")
    for edge_type, count in sorted(graph_metrics["edges_by_type"].items()):
        print(f"  • {edge_type}: {count:,}")
    
    # Precision/Recall/F1 metrics
    if "precision_recall_f1" in metrics:
        prf = metrics["precision_recall_f1"]
        print("\n📈 PRECISION, RECALL, AND F1 SCORES:")
        
        # Schema compliance
        schema = prf["schema_compliance"]
        print("\n  Schema Compliance:")
        print(f"    ⚠ Note: These metrics measure SCHEMA COMPLIANCE (if expected types exist),")
        print(f"      not extraction quality. Perfect scores (1.0) mean all expected types")
        print(f"      are present, but don't indicate if entities/relations are correctly extracted.")
        print(f"\n    Node Types:")
        print(f"      - Precision: {schema['node_types']['precision']:.4f} (extracted types that match schema)")
        print(f"      - Recall: {schema['node_types']['recall']:.4f} (expected types that were extracted)")
        print(f"      - F1 Score: {schema['node_types']['f1']:.4f}")
        if "coverage" in schema:
            print(f"      - Coverage: {dict(schema['coverage']['node_types'])}")
        print(f"\n    Edge Types:")
        print(f"      - Precision: {schema['edge_types']['precision']:.4f} (extracted types that match schema)")
        print(f"      - Recall: {schema['edge_types']['recall']:.4f} (expected types that were extracted)")
        print(f"      - F1 Score: {schema['edge_types']['f1']:.4f}")
        if "coverage" in schema:
            print(f"      - Coverage: {dict(schema['coverage']['edge_types'])}")
        
        # Completeness
        completeness = prf["completeness"]
        print("\n  Relation Completeness:")
        print(f"    - Speaker Relation: {completeness['speaker_relation']:.4f} ({completeness['speaker_relation']*100:.1f}%)")
        print(f"    - Date Relation: {completeness['date_relation']:.4f} ({completeness['date_relation']*100:.1f}%)")
        print(f"    - Topic Relation: {completeness['topic_relation']:.4f} ({completeness['topic_relation']*100:.1f}%)")
        print(f"    - Overall Completeness: {completeness['overall']:.4f} ({completeness['overall']*100:.1f}%)")
        
        # Accuracy
        accuracy = prf["accuracy"]
        print(f"\n  Accuracy:")
        print(f"    - Date Parsing: {accuracy['date_parsing']:.4f} ({accuracy['date_parsing']*100:.1f}%)")
        
        # Macro F1
        print(f"\n  Overall Macro F1: {prf['macro_f1']:.4f}")
    
    # Entity resolution metrics
    if "entity_resolution" in metrics:
        er = metrics["entity_resolution"]
        print("\n🎯 ENTITY RESOLUTION QUALITY:")
        print(f"  Speakers:")
        print(f"    - Unique Entities: {er['speakers']['unique_entities']:,}")
        print(f"    - Total Nodes: {er['speakers']['total_nodes']:,}")
        print(f"    - Resolution Ratio: {er['speakers']['resolution_ratio']:.4f}")
        print(f"  Topics:")
        print(f"    - Unique Entities: {er['topics']['unique_entities']:,}")
        print(f"    - Total Nodes: {er['topics']['total_nodes']:,}")
        print(f"    - Resolution Ratio: {er['topics']['resolution_ratio']:.4f}")
    
    # Topic quality metrics
    topic_metrics = metrics["topics"]
    print("\n📚 TOPIC QUALITY METRICS:")
    print(f"  • Unique Topic Nodes: {topic_metrics['topic_nodes']:,}")
    print(f"  • Average Intra-Cluster Similarity: {topic_metrics['avg_intra_similarity']:.4f}")
    print(f"  • Average Inter-Cluster Similarity: {topic_metrics['avg_inter_similarity']:.4f}")
    print(f"  • Coherence Ratio (intra/inter): {topic_metrics.get('coherence_ratio', 0.0):.4f}")
    if topic_metrics.get('coherence_ratio', 0) > 1.0:
        print("    ✓ Topics are well-clustered (intra > inter similarity)")
    else:
        print("    ⚠ Topics may need better clustering (intra ≤ inter similarity)")
    
    # Date parsing stats
    date_stats = metrics["dates"]
    print("\n📅 DATE PARSING STATISTICS:")
    total_dates = date_stats.get("success", 0) + date_stats.get("failure", 0)
    if total_dates > 0:
        success_rate = date_stats.get("success", 0) / total_dates * 100
        print(f"  • Successfully Parsed: {date_stats.get('success', 0):,} ({success_rate:.1f}%)")
        print(f"  • Failed to Parse: {date_stats.get('failure', 0):,} ({100-success_rate:.1f}%)")
    
    # Abbreviation stats
    if "abbreviations" in metrics:
        abbr_stats = metrics["abbreviations"]
        print("\n🔤 ABBREVIATION EXPANSION:")
        print(f"  • Total Expansions: {abbr_stats.get('total_expansions', 0):,}")
        print(f"  • Unique Abbreviations Found: {abbr_stats.get('unique_abbreviations', 0):,}")
    
    print("\n" + "="*80)
    print(f"Full metrics saved to: {metrics_path}")
    print("="*80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ECB knowledge graph.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the ECB speeches CSV.",
    )
    parser.add_argument(
        "--delimiter",
        default="|",
        help="CSV delimiter (default: |).",
    )
    parser.add_argument(
        "--output-dir",
        default="kg_outputs",
        help="Directory where nodes/edges CSVs will be written.",
    )
    parser.add_argument(
        "--abbreviation-config",
        default=None,
        help="Optional YAML file with additional abbreviation expansions.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model name.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=3,
        help="Maximum topics to attach per speech.",
    )
    parser.add_argument(
        "--topic-cluster-threshold",
        type=float,
        default=0.72,
        help="Similarity threshold when clustering phrases within a speech.",
    )
    parser.add_argument(
        "--topic-merge-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold when merging global topic nodes.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional limit for quicker experimentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_pipeline(parse_args())

