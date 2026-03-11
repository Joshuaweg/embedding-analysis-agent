#!/usr/bin/env python3
"""
value_analysis.py -- Research script: Do GPT-2 embeddings encode moral value structure?

Runs 4 experiments on the precomputed Mapper graph:
  1. Moral pole separation (graph topology)
  2. Community coherence of MFT categories (graph topology)
  3. WEAT baseline on raw GPT-2 embeddings
  4. Value axis projection across full vocabulary

Usage:
    python value_analysis.py                    # full run
    python value_analysis.py --fast             # reduced sampling for validation
    python value_analysis.py --graph PATH       # custom graph file
    python value_analysis.py --skip-embeddings  # run only experiments 1-2
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from graph_structure import TokenGraph
from value_lexicons import (
    MORAL_FOUNDATIONS,
    SCHWARTZ_VALUES,
    get_mft_pole_tokens,
)

DEFAULT_GRAPH = "node_clusters_with_weights.json"


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(graph_path: str) -> TokenGraph:
    """Load the token graph from a JSON file, including edge weights."""
    graph = TokenGraph.from_json(graph_path)
    with open(graph_path) as f:
        data = json.load(f)
    graph.edge_weights = {}
    for k, v in data.get("edge_weights", {}).items():
        a, b = k.split("|")
        graph.edge_weights[(a, b)] = v
    return graph


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_gpt2_embeddings() -> "np.ndarray | None":
    """Load GPT-2 token embedding matrix (50257 x 768). Returns None if unavailable.

    Matches the approach used in embedding.py: GPT2Model.from_pretrained + PyTorch.
    """
    try:
        import torch
        from transformers import GPT2Model
        print("Loading GPT-2 model weights...")
        model = GPT2Model.from_pretrained("gpt2")
        matrix = model.get_input_embeddings().weight.detach().numpy()
        print(f"Loaded embedding matrix: {matrix.shape}")
        return matrix
    except Exception as e:
        print(f"Could not load GPT-2 embeddings: {e}")
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between all pairs of rows in A and B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


# ---------------------------------------------------------------------------
# Experiment 1 -- Moral Pole Separation
# ---------------------------------------------------------------------------

def experiment_pole_separation(graph: TokenGraph, fast: bool = False) -> list[dict]:
    """For each MFT foundation, measure BFS distances within and between poles."""
    print("\n=== Experiment 1: Moral Pole Separation ===")
    results = []
    max_samples = 3 if fast else 5

    all_node_ids = list(graph.nodes.keys())

    for name, foundation in MORAL_FOUNDATIONS.items():
        print(f"  Processing {name}...")

        pos_tokens = foundation["positive"]
        neg_tokens = foundation["negative"]

        pos_nodes = []
        for t in pos_tokens:
            pos_nodes.extend(graph.find_nodes_with_token(t))
        neg_nodes = []
        for t in neg_tokens:
            neg_nodes.extend(graph.find_nodes_with_token(t))

        pos_ids = list({n.id for n in pos_nodes})[:max_samples]
        neg_ids = list({n.id for n in neg_nodes})[:max_samples]

        if len(pos_ids) < 2 or len(neg_ids) < 2:
            results.append({
                "foundation": name,
                "status": "insufficient_nodes",
                "positive_nodes": len(pos_ids),
                "negative_nodes": len(neg_ids),
            })
            continue

        # Intra-positive distances
        intra_pos = []
        for i, a in enumerate(pos_ids):
            for b in pos_ids[i + 1:]:
                path = graph.bfs_path(a, b)
                if path:
                    intra_pos.append(len(path) - 1)

        # Intra-negative distances
        intra_neg = []
        for i, a in enumerate(neg_ids):
            for b in neg_ids[i + 1:]:
                path = graph.bfs_path(a, b)
                if path:
                    intra_neg.append(len(path) - 1)

        # Inter-pole distances
        inter = []
        for a in pos_ids:
            for b in neg_ids:
                path = graph.bfs_path(a, b)
                if path:
                    inter.append(len(path) - 1)

        mean_intra_pos = np.mean(intra_pos) if intra_pos else float("nan")
        mean_intra_neg = np.mean(intra_neg) if intra_neg else float("nan")
        mean_inter = np.mean(inter) if inter else float("nan")
        mean_intra = np.nanmean([mean_intra_pos, mean_intra_neg])
        separation_ratio = float(mean_inter / mean_intra) if mean_intra > 0 else float("nan")

        # Null baseline: random pairs
        null_distances = []
        for _ in range(len(inter) if inter else 10):
            a, b = random.sample(all_node_ids, 2)
            path = graph.bfs_path(a, b)
            if path:
                null_distances.append(len(path) - 1)

        results.append({
            "foundation": name,
            "positive_nodes": len(pos_ids),
            "negative_nodes": len(neg_ids),
            "mean_intra_positive": round(float(mean_intra_pos), 2) if not np.isnan(mean_intra_pos) else None,
            "mean_intra_negative": round(float(mean_intra_neg), 2) if not np.isnan(mean_intra_neg) else None,
            "mean_inter_pole": round(float(mean_inter), 2) if not np.isnan(mean_inter) else None,
            "separation_ratio": round(separation_ratio, 3) if not np.isnan(separation_ratio) else None,
            "null_baseline_mean": round(float(np.mean(null_distances)), 2) if null_distances else None,
        })
        print(f"    separation_ratio={results[-1]['separation_ratio']}, "
              f"null_baseline={results[-1]['null_baseline_mean']}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2 -- Community Coherence
# ---------------------------------------------------------------------------

def experiment_community_coherence(graph: TokenGraph, fast: bool = False) -> list[dict]:
    """Check if MFT tokens cluster into the same graph communities."""
    print("\n=== Experiment 2: Community Coherence ===")

    community_result = graph.detect_communities()
    node_to_community: dict[str, int] = {}
    for comm_idx, comm_set in enumerate(community_result["communities"]):
        for node_id in comm_set:
            node_to_community[node_id] = comm_idx

    all_node_ids = list(graph.nodes.keys())
    results = []
    null_samples = 20 if fast else 100

    for name, foundation in MORAL_FOUNDATIONS.items():
        for pole in ("positive", "negative"):
            tokens = foundation[pole]
            pole_nodes = []
            for t in tokens:
                pole_nodes.extend(graph.find_nodes_with_token(t))
            pole_ids = list({n.id for n in pole_nodes})

            if not pole_ids:
                results.append({
                    "foundation": name,
                    "pole": pole,
                    "status": "no_nodes",
                })
                continue

            communities = [node_to_community.get(nid) for nid in pole_ids]
            communities = [c for c in communities if c is not None]

            if not communities:
                results.append({
                    "foundation": name,
                    "pole": pole,
                    "status": "no_community_mapping",
                })
                continue

            unique_communities = set(communities)
            community_counts = Counter(communities)
            dominant_count = community_counts.most_common(1)[0][1]
            pct_dominant = dominant_count / len(communities)
            cohesion = 1.0 / len(unique_communities)

            # Null baseline: random samples of same size
            null_cohesions = []
            for _ in range(null_samples):
                sample = random.sample(all_node_ids, min(len(pole_ids), len(all_node_ids)))
                sample_comms = [node_to_community.get(nid) for nid in sample]
                sample_comms = [c for c in sample_comms if c is not None]
                if sample_comms:
                    null_cohesions.append(1.0 / len(set(sample_comms)))

            entry = {
                "foundation": name,
                "pole": pole,
                "num_nodes": len(pole_ids),
                "unique_communities": len(unique_communities),
                "cohesion": round(cohesion, 4),
                "pct_dominant_community": round(pct_dominant, 4),
                "null_baseline_cohesion": round(float(np.mean(null_cohesions)), 4) if null_cohesions else None,
            }
            results.append(entry)
            print(f"  {name}.{pole}: cohesion={entry['cohesion']}, "
                  f"pct_dominant={entry['pct_dominant_community']}, "
                  f"null={entry['null_baseline_cohesion']}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3 -- WEAT on Raw Embeddings
# ---------------------------------------------------------------------------

WEAT_SETS = {
    "male_names": ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"],
    "female_names": ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"],
    "european_names": ["Adam", "Harry", "Josh", "Roger", "Alan", "Frank", "Ryan", "Andrew"],
    "african_names": ["Alonzo", "Jamel", "Theo", "Jerome", "Leroy", "Lamar", "Lionel", "Deion"],
    "pleasant": ["joy", "love", "peace", "wonderful", "pleasure", "beautiful", "happy", "glorious"],
    "unpleasant": ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"],
}


def weat_effect_size(
    embeddings: np.ndarray,
    target_a_ids: list[int],
    target_b_ids: list[int],
    attr_x_ids: list[int],
    attr_y_ids: list[int],
) -> float:
    """Compute WEAT effect size (Cohen's d) between two target sets and two attribute sets."""
    if not target_a_ids or not target_b_ids or not attr_x_ids or not attr_y_ids:
        return float("nan")

    emb_a = embeddings[target_a_ids]
    emb_b = embeddings[target_b_ids]
    emb_x = embeddings[attr_x_ids]
    emb_y = embeddings[attr_y_ids]

    def association(target_emb: np.ndarray) -> np.ndarray:
        sim_x = cosine_sim_matrix(target_emb, emb_x).mean(axis=1)
        sim_y = cosine_sim_matrix(target_emb, emb_y).mean(axis=1)
        return sim_x - sim_y

    assoc_a = association(emb_a)
    assoc_b = association(emb_b)

    mean_diff = assoc_a.mean() - assoc_b.mean()
    all_assoc = np.concatenate([assoc_a, assoc_b])
    std = all_assoc.std()
    if std == 0:
        return float("nan")
    return float(mean_diff / std)


def resolve_token_ids(tokenizer, tokens: list[str]) -> list[int]:
    """Resolve a list of token strings to GPT-2 token IDs. Tries bare and space-prefixed forms."""
    ids = []
    for t in tokens:
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(encoded[0])
        else:
            # Try space-prefixed
            encoded2 = tokenizer.encode(" " + t, add_special_tokens=False)
            if len(encoded2) == 1:
                ids.append(encoded2[0])
    return ids


def experiment_weat(embeddings: np.ndarray, fast: bool = False) -> list[dict]:
    """Run WEAT tests using GPT-2 embeddings."""
    print("\n=== Experiment 3: WEAT on Raw Embeddings ===")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    results = []

    # Resolve WEAT standard sets
    resolved = {}
    for name, tokens in WEAT_SETS.items():
        resolved[name] = resolve_token_ids(tokenizer, tokens)

    # Resolve MFT sets
    mft_resolved = {}
    for fname in ("care_harm", "fairness_cheating"):
        for pole in ("positive", "negative"):
            key = f"{fname}_{pole}"
            bare_tokens = [t for t in MORAL_FOUNDATIONS[fname][pole] if not t.startswith(" ")]
            mft_resolved[key] = resolve_token_ids(tokenizer, bare_tokens)

    # Standard WEAT: gender + pleasant/unpleasant
    d = weat_effect_size(
        embeddings,
        resolved["male_names"], resolved["female_names"],
        resolved["pleasant"], resolved["unpleasant"],
    )
    results.append({
        "test": "gender_pleasant",
        "targets": "male_names vs female_names",
        "attributes": "pleasant vs unpleasant",
        "effect_size_d": round(d, 4) if not np.isnan(d) else None,
    })
    print(f"  gender_pleasant: d={results[-1]['effect_size_d']}")

    # Standard WEAT: race + pleasant/unpleasant
    d = weat_effect_size(
        embeddings,
        resolved["european_names"], resolved["african_names"],
        resolved["pleasant"], resolved["unpleasant"],
    )
    results.append({
        "test": "race_pleasant",
        "targets": "european_names vs african_names",
        "attributes": "pleasant vs unpleasant",
        "effect_size_d": round(d, 4) if not np.isnan(d) else None,
    })
    print(f"  race_pleasant: d={results[-1]['effect_size_d']}")

    # MFT-extended: gender + care/harm
    d = weat_effect_size(
        embeddings,
        resolved["male_names"], resolved["female_names"],
        mft_resolved.get("care_harm_positive", []),
        mft_resolved.get("care_harm_negative", []),
    )
    results.append({
        "test": "gender_care_harm",
        "targets": "male_names vs female_names",
        "attributes": "care_positive vs harm_negative",
        "effect_size_d": round(d, 4) if not np.isnan(d) else None,
    })
    print(f"  gender_care_harm: d={results[-1]['effect_size_d']}")

    # MFT-extended: gender + fairness/cheating
    d = weat_effect_size(
        embeddings,
        resolved["male_names"], resolved["female_names"],
        mft_resolved.get("fairness_cheating_positive", []),
        mft_resolved.get("fairness_cheating_negative", []),
    )
    results.append({
        "test": "gender_fairness_cheating",
        "targets": "male_names vs female_names",
        "attributes": "fairness_positive vs cheating_negative",
        "effect_size_d": round(d, 4) if not np.isnan(d) else None,
    })
    print(f"  gender_fairness_cheating: d={results[-1]['effect_size_d']}")

    return results


# ---------------------------------------------------------------------------
# Experiment 4 -- Value Axis Projection
# ---------------------------------------------------------------------------

def build_value_axis(
    embeddings: np.ndarray,
    pos_tokens: list[str],
    neg_tokens: list[str],
    tokenizer,
) -> "np.ndarray | None":
    """Construct a moral axis vector from positive and negative pole token embeddings."""
    pos_ids = resolve_token_ids(tokenizer, [t for t in pos_tokens if not t.startswith(" ")])
    neg_ids = resolve_token_ids(tokenizer, [t for t in neg_tokens if not t.startswith(" ")])

    if not pos_ids or not neg_ids:
        return None

    pos_mean = embeddings[pos_ids].mean(axis=0)
    neg_mean = embeddings[neg_ids].mean(axis=0)
    axis = pos_mean - neg_mean
    norm = np.linalg.norm(axis)
    if norm == 0:
        return None
    return axis / norm


def experiment_value_projection(embeddings: np.ndarray, fast: bool = False) -> list[dict]:
    """Project all embeddings onto MFT value axes."""
    print("\n=== Experiment 4: Value Axis Projection ===")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    results = []
    top_k = 10 if fast else 20

    for name, foundation in MORAL_FOUNDATIONS.items():
        axis = build_value_axis(
            embeddings,
            foundation["positive"],
            foundation["negative"],
            tokenizer,
        )
        if axis is None:
            results.append({"foundation": name, "status": "axis_construction_failed"})
            continue

        # Project all embeddings
        projections = embeddings @ axis

        # Top positive and negative
        top_pos_idx = np.argsort(projections)[-top_k:][::-1]
        top_neg_idx = np.argsort(projections)[:top_k]

        top_pos_tokens = []
        for idx in top_pos_idx:
            tok = tokenizer.decode([int(idx)])
            top_pos_tokens.append({"token": tok, "projection": round(float(projections[idx]), 4)})

        top_neg_tokens = []
        for idx in top_neg_idx:
            tok = tokenizer.decode([int(idx)])
            top_neg_tokens.append({"token": tok, "projection": round(float(projections[idx]), 4)})

        # Demographic targets
        demographic_scores = {}
        for demo_name, demo_tokens in WEAT_SETS.items():
            demo_ids = resolve_token_ids(tokenizer, demo_tokens)
            if demo_ids:
                demographic_scores[demo_name] = round(float(projections[demo_ids].mean()), 4)

        entry = {
            "foundation": name,
            "top_positive": top_pos_tokens,
            "top_negative": top_neg_tokens,
            "demographic_projections": demographic_scores,
        }
        results.append(entry)
        print(f"  {name}: top_pos={top_pos_tokens[0]['token']!r}, "
              f"top_neg={top_neg_tokens[0]['token']!r}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research script: Do GPT-2 embeddings encode moral value structure?",
    )
    parser.add_argument(
        "--graph", default=DEFAULT_GRAPH,
        help=f"Path to graph JSON file (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced sampling for quick validation",
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip experiments 3 and 4 (no embedding download needed)",
    )
    args = parser.parse_args()

    # Load graph
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"Error: graph file not found: {graph_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading graph from {graph_path}...")
    graph = load_graph(str(graph_path))
    print(f"Graph loaded: {len(graph.nodes)} nodes, "
          f"{sum(len(n.connected_nodes) for n in graph.nodes.values()) // 2} edges")

    results: dict = {
        "metadata": {
            "graph_nodes": len(graph.nodes),
            "graph_edges": sum(len(n.connected_nodes) for n in graph.nodes.values()) // 2,
            "run_date": datetime.now().isoformat(),
            "fast_mode": args.fast,
        },
    }

    # Experiment 1
    results["experiment_1_pole_separation"] = experiment_pole_separation(graph, args.fast)

    # Experiment 2
    results["experiment_2_community_coherence"] = experiment_community_coherence(graph, args.fast)

    # Experiments 3 and 4
    if args.skip_embeddings:
        results["experiment_3_weat"] = "skipped: --skip-embeddings flag"
        results["experiment_4_value_projection"] = "skipped: --skip-embeddings flag"
    else:
        print("\nLoading GPT-2 embeddings...")
        embeddings = load_gpt2_embeddings()
        if embeddings is None:
            print("WARNING: Could not load GPT-2 embeddings. Skipping experiments 3 and 4.")
            results["experiment_3_weat"] = "skipped: embeddings unavailable"
            results["experiment_4_value_projection"] = "skipped: embeddings unavailable"
        else:
            print(f"Embeddings loaded: shape {embeddings.shape}")
            results["experiment_3_weat"] = experiment_weat(embeddings, args.fast)
            results["experiment_4_value_projection"] = experiment_value_projection(embeddings, args.fast)

    # Write output
    out_dir = Path("_meta/research")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "value_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results written to {json_path}")

    txt_path = out_dir / "value_analysis_results.txt"
    with open(txt_path, "w") as f:
        f.write("Value Analysis Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {results['metadata']['run_date']}\n")
        f.write(f"Graph: {results['metadata']['graph_nodes']} nodes, "
                f"{results['metadata']['graph_edges']} edges\n")
        f.write(f"Fast mode: {results['metadata']['fast_mode']}\n\n")

        f.write("Experiment 1: Moral Pole Separation\n")
        f.write("-" * 40 + "\n")
        if isinstance(results["experiment_1_pole_separation"], list):
            for entry in results["experiment_1_pole_separation"]:
                f.write(f"  {entry.get('foundation', 'unknown')}: "
                        f"separation_ratio={entry.get('separation_ratio')}, "
                        f"null_baseline={entry.get('null_baseline_mean')}\n")

        f.write("\nExperiment 2: Community Coherence\n")
        f.write("-" * 40 + "\n")
        if isinstance(results["experiment_2_community_coherence"], list):
            for entry in results["experiment_2_community_coherence"]:
                f.write(f"  {entry.get('foundation', 'unknown')}.{entry.get('pole', '?')}: "
                        f"cohesion={entry.get('cohesion')}, "
                        f"pct_dominant={entry.get('pct_dominant_community')}\n")

        f.write("\nExperiment 3: WEAT\n")
        f.write("-" * 40 + "\n")
        exp3 = results["experiment_3_weat"]
        if isinstance(exp3, list):
            for entry in exp3:
                f.write(f"  {entry.get('test', 'unknown')}: d={entry.get('effect_size_d')}\n")
        else:
            f.write(f"  {exp3}\n")

        f.write("\nExperiment 4: Value Axis Projection\n")
        f.write("-" * 40 + "\n")
        exp4 = results["experiment_4_value_projection"]
        if isinstance(exp4, list):
            for entry in exp4:
                fname = entry.get("foundation", "unknown")
                if "top_positive" in entry:
                    top = entry["top_positive"][0]["token"] if entry["top_positive"] else "N/A"
                    bot = entry["top_negative"][0]["token"] if entry["top_negative"] else "N/A"
                    f.write(f"  {fname}: most_positive={top!r}, most_negative={bot!r}\n")
                else:
                    f.write(f"  {fname}: {entry.get('status', 'unknown')}\n")
        else:
            f.write(f"  {exp4}\n")

    print(f"Text results written to {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
