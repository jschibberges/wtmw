
from __future__ import annotations
import numpy as np
import json
import pandas as pd
from datetime import datetime
import networkx as nx
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, List, Any
from functools import lru_cache

from scrape_talkshows import load_json_file, save_json_file
from nlp_utils import clean_guest_rows, clean_description_for_labels, _dedupe_topic_terms, _filter_topic_terms, get_german_stopwords
from guest_classification import add_classification_columns
from guest_classification_embedding import build_prototype_embeddings, apply_embedding_fallback

try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except Exception:
    Network = None
    _PYVIS_AVAILABLE = False



# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def clean_str(text):
    if text is not None:
        text = text.strip()
        if "| Bild: " in text:
            text = ""
    else:
        text=""
    return text

@lru_cache(maxsize=1)
def _load_name_corrections() -> dict:
    """Loads name corrections from JSON file. Cached to avoid repeated file I/O."""
    corrections_path = DATA_DIR / "name_corrections.json"
    if not corrections_path.exists():
        print("Warning: name_corrections.json not found. Skipping manual corrections.")
        return {}

    name_corrections = load_json_file(corrections_path)
    # Ensure that the loaded data is a dictionary
    if isinstance(name_corrections, dict):
        return name_corrections

    # If not a dict (e.g., file was empty or malformed), return empty dict
    return {}

def manual_name_corrections(name):
    """Applies manual name corrections from a cached mapping file."""
    corrections = _load_name_corrections()
    return corrections.get(name, name)

def prepare_guest_dataframe(all_data):
    """Extracts guests from all shows and prepares a DataFrame."""
    guest_list = []
    for dat in all_data:
        # Ensure episode has guests and a UID for linking
        if "guests" in dat and dat.get("guests") and "uid" in dat:
            for guest in dat["guests"]:
                # Ensure guest is a dictionary and has a name
                if isinstance(guest, dict) and guest.get("name"):
                    dic = {}
                    dic.update(guest)
                    dic["uid"] = dat.get("uid")
                    dic["Talkshow"] = f"{dat.get('show', 'N/A')} - {dat.get('date', 'N/A')}"
                    guest_list.append(dic)

    if not guest_list:
        return pd.DataFrame()

    df_guests = pd.DataFrame(guest_list)
    
    # Deduplicate guests within the same episode (if scraped multiple times)
    df_guests.drop_duplicates(subset=['name', 'uid'], inplace=True)

    df_guests['name'] = df_guests['name'].apply(manual_name_corrections)
    df_guests['role'] = df_guests['role'].apply(clean_str)
    df_guests['description'] = df_guests['description'].apply(clean_str)
    return df_guests

def consolidate_guests(df_cleaned_guests: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates guest data to one row per unique name.

    Args:
        df_cleaned_guests: DataFrame of cleaned guest data, typically from `clean_guest_rows`.

    Returns:
        A DataFrame with one row per unique guest.
    """
    def consolidate_party(parties):
        """Selects the most representative party from a list."""
        unique_parties = {p for p in parties if pd.notna(p)}
        if not unique_parties: return None
        if len(unique_parties) == 1: return unique_parties.pop()

        # Prioritize specific parties if present
        if "BSW" in unique_parties:
            return "BSW"
        if "parteilos" in unique_parties:
            return "parteilos"

        # Fallback to the most frequent party
        mode_series = pd.Series(list(parties)).mode()
        return mode_series.iloc[0] if not mode_series.empty else None

    # Group by the cleaned name and aggregate other columns
    df_consolidated = df_cleaned_guests.groupby("name_clean").agg(
        party_norm=('party_norm', consolidate_party),
        role_clean=('role_clean', lambda x: list(x.dropna().unique())),
        description=('description', lambda x: list(x.dropna().unique())),
        uid=('uid', lambda x: list(x.dropna().unique())),
        Talkshow=('Talkshow', lambda x: list(x.dropna().unique())),
        Count=('uid', 'nunique')  # Add count of unique appearances
    ).reset_index()

    return df_consolidated


def _prepare_static_network_view(
    G: nx.Graph,
    *,
    min_edge_weight: int = 2,
    max_nodes: int = 120,
) -> nx.Graph:
    """Reduce a dense guest graph to a readable static subset."""
    if G.number_of_nodes() == 0:
        return nx.Graph()

    def _graph_for_threshold(threshold: float) -> nx.Graph:
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        H.add_edges_from(
            (u, v, data)
            for u, v, data in G.edges(data=True)
            if float(data.get("weight", 1)) >= float(threshold)
        )
        H.remove_nodes_from(list(nx.isolates(H)))
        return H

    edge_weights = sorted(
        {
            float(data.get("weight", 1))
            for _, _, data in G.edges(data=True)
            if float(data.get("weight", 1)) >= float(min_edge_weight)
        }
    )

    if edge_weights:
        H = _graph_for_threshold(edge_weights[0])
        for threshold in edge_weights[1:]:
            if H.number_of_nodes() <= max_nodes:
                break
            candidate = _graph_for_threshold(threshold)
            if candidate.number_of_nodes() == 0:
                break
            H = candidate
    else:
        H = G.copy()

    if H.number_of_edges() > 0:
        largest_component = max(nx.connected_components(H), key=len)
        H = H.subgraph(largest_component).copy()

    if H.number_of_nodes() > max_nodes:
        ranked_nodes = sorted(
            H.nodes(),
            key=lambda node: (
                -float(H.nodes[node].get("appearances", 0)),
                -float(H.degree(node, weight="weight")),
                str(node),
            ),
        )
        H = H.subgraph(ranked_nodes[:max_nodes]).copy()

    return H


def _detect_graph_communities(G: nx.Graph) -> dict[str, int]:
    """Assign each node to a community for coloring."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {node: idx for idx, node in enumerate(G.nodes())}

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    mapping: dict[str, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            mapping[node] = idx
    return mapping


def _select_label_nodes(G: nx.Graph, *, top_n: int = 30) -> list[str]:
    """Return the most important nodes to label in a static plot."""
    ranked = sorted(
        G.nodes(),
        key=lambda node: (
            -float(G.nodes[node].get("appearances", 0)),
            -float(G.degree(node, weight="weight")),
            str(node),
        ),
    )
    return ranked[:top_n]

def visualize_network(
    G,
    df_guests=None,
    layout="spring",
    seed=42,
    figsize=(12, 9),
    output_path: Optional[Path | str] = None,
    min_edge_weight: int = 2,
    max_nodes: int = 120,
    label_top_n: int = 18,
):
    """
    Draw a curated static co-occurrence graph.

    Args:
        G: networkx.Graph with optional node attr 'topic_range'
        df_guests: (optional) DataFrame for tooltips/labels (not required here)
        layout: 'spring' | 'kamada_kawai' | 'fr' | 'spectral' | 'circular'
        seed: random seed for layouts that support it
        figsize: figure size
    """
    H = _prepare_static_network_view(G, min_edge_weight=min_edge_weight, max_nodes=max_nodes)
    if H.number_of_nodes() == 0:
        print("Guest network empty after filtering; skipping static visualization.")
        return

    # ---- positions
    if layout == "spring":
        spring_k = max(0.35, 2.2 / np.sqrt(max(H.number_of_nodes(), 1)))
        pos = nx.spring_layout(H, seed=seed, k=spring_k, iterations=300, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H, weight="weight")
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(H, seed=seed)
    elif layout == "spectral":
        pos = nx.spectral_layout(H)
    elif layout == "circular":
        pos = nx.circular_layout(H)
    else:
        pos = nx.spring_layout(H, seed=seed, weight="weight")

    # ---- node sizes: emphasize frequent guests
    appearances = nx.get_node_attributes(H, "appearances")
    deg = dict(H.degree())
    node_sizes = [160 + 35 * np.sqrt(max(appearances.get(n, deg.get(n, 1)), 1)) for n in H.nodes()]

    # ---- node colors by community for a cleaner static view
    communities = _detect_graph_communities(H)
    community_cmap = mpl.colormaps.get_cmap("tab20").resampled(
        max(len(set(communities.values())), 1)
    )
    node_colors = [community_cmap(communities.get(node, 0)) for node in H.nodes()]

    # ---- draw
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    edge_widths = [0.35 + 0.45 * float(data.get("weight", 1)) for _, _, data in H.edges(data=True)]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.18, edge_color="#7f8c8d", width=edge_widths)

    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.7,
        edgecolors="white",
        alpha=0.95,
    )

    label_nodes = _select_label_nodes(H, top_n=label_top_n)
    label_pos = {node: pos[node] for node in label_nodes if node in pos}
    nx.draw_networkx_labels(
        H,
        label_pos,
        labels={node: node for node in label_nodes},
        ax=ax,
        font_size=8,
        font_weight="medium",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.15},
    )

    fig.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"Guest network saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)

def export_interactive_network(
    G: nx.Graph,
    filepath: Path | str,
    *,
    value_attr: str | None = None,
    tooltip_labels: Dict[str, str] | None = None,
    physics: bool = True,
    height: str = "800px",
    width: str = "100%",
    min_edge_weight: int | None = None,
    max_nodes: int | None = None,
    label_top_n: int | None = None,
    community_colors: bool = False,
) -> None:
    """Save an interactive PyVis network if the dependency is available."""
    if not _PYVIS_AVAILABLE or Network is None:
        print(f"PyVis not available. Skipping interactive export for {filepath}.")
        return

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    H = G
    if min_edge_weight is not None or max_nodes is not None:
        H = _prepare_static_network_view(
            G,
            min_edge_weight=min_edge_weight or 1,
            max_nodes=max_nodes or G.number_of_nodes(),
        )

    net = Network(height=height, width=width, bgcolor="#ffffff", font_color="#2b2b2b", directed=H.is_directed())
    if physics:
        net.barnes_hut()
    else:
        net.toggle_physics(False)

    net.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true
          },
          "nodes": {
            "shape": "dot",
            "font": {
              "size": 18,
              "face": "Arial"
            },
            "scaling": {
              "min": 10,
              "max": 40
            }
          },
          "edges": {
            "smooth": false,
            "color": {
              "inherit": false,
              "color": "#c7d1db",
              "highlight": "#7f8c8d"
            },
            "scaling": {
              "min": 1,
              "max": 8
            }
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -3500,
              "centralGravity": 0.12,
              "springLength": 165,
              "springConstant": 0.02,
              "damping": 0.88,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75
          }
        }
        """
    )

    tooltip_labels = tooltip_labels or {}
    label_nodes = set(_select_label_nodes(H, top_n=label_top_n or H.number_of_nodes()))
    communities = _detect_graph_communities(H) if community_colors else {}
    if communities:
        community_cmap = mpl.colormaps.get_cmap("tab20").resampled(
            max(len(set(communities.values())), 1)
        )
    else:
        community_cmap = None

    for node, data in H.nodes(data=True):
        tooltip_lines = [str(node)]
        for key, label in tooltip_labels.items():
            val = data.get(key)
            if val is None or val == "":
                continue
            tooltip_lines.append(f"{label}: {val}")

        node_value = None
        if value_attr:
            val = data.get(value_attr)
            if isinstance(val, (int, float)) and val > 0:
                node_value = float(val)

        if node_value is None:
            deg = H.degree(node)
            node_value = float(deg if deg > 0 else 1.0)

        color = None
        if community_cmap is not None:
            rgba = community_cmap(communities.get(node, 0))
            color = mpl.colors.to_hex(rgba)

        net.add_node(
            node,
            label=str(node) if node in label_nodes else "",
            title="<br>".join(tooltip_lines),
            value=node_value,
            color=color,
        )

    for source, target, edge_data in H.edges(data=True):
        weight = edge_data.get("weight", 1)
        try:
            weight_val = float(weight)
            if weight_val <= 0:
                weight_val = 1.0
        except Exception:
            weight_val = 1.0
        net.add_edge(source, target, value=weight_val, title=f"Gewicht: {weight}")

    net.write_html(str(path), notebook=False)
    print(f"Interactive network saved to {path}")


def analyze_topic_guest_connections(df_guests_with_topics):
    """
    Analyzes the connections between guests and topics.
    - Calculates topic range for each guest.
    - Creates a co-occurrence network of topics based on shared guests.
    """
    print("\n--- Analyzing Guest-Topic Connections ---")

    # Filter out outlier topics and rows with no topic assigned
    valid_topics_df = df_guests_with_topics.dropna(subset=['topic_label'])
    valid_topics_df = valid_topics_df[valid_topics_df['topic'] != -1]

    if valid_topics_df.empty:
        print("No valid guest-topic connections found to analyze.")
        return

    # 1. Calculate and display guest "topic range"
    guest_topic_range = valid_topics_df.groupby('name')['topic_label'].nunique().sort_values(ascending=False)
    print("\n--- Top 15 Guests by Topic Range ---")
    print(guest_topic_range.head(15))
    guest_topic_range.to_excel(DATA_DIR / "guest_topic_range.xlsx")
    print(f"Guest topic range saved to {DATA_DIR / 'guest_topic_range.xlsx'}")

    # 2. Analyze which topics share the most guests (Topic Co-occurrence)
    topic_guest_list = valid_topics_df.groupby('topic_label')['name'].apply(lambda x: list(set(x))).reset_index()

    T = nx.Graph()
    for index, row in topic_guest_list.iterrows():
        # Add node with size attribute based on number of unique guests
        T.add_node(row['topic_label'], num_guests=len(row['name']))

    # Create edges based on shared guests
    for i in range(len(topic_guest_list)):
        for j in range(i + 1, len(topic_guest_list)):
            topic1_guests = set(topic_guest_list.iloc[i]['name'])
            topic2_guests = set(topic_guest_list.iloc[j]['name'])
            shared_guests = topic1_guests.intersection(topic2_guests)
            
            if len(shared_guests) > 1: # Only connect topics with more than 1 shared guest
                T.add_edge(topic_guest_list.iloc[i]['topic_label'], topic_guest_list.iloc[j]['topic_label'], weight=len(shared_guests))

    print(f"\nCreated Topic Co-occurrence Network with {T.number_of_nodes()} topics and {T.number_of_edges()} connections.")

    # Visualize the topic network
    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(T, k=0.8, iterations=50)
    node_sizes = [data['num_guests'] * 25 + 100 for _, data in T.nodes(data=True)]
    edge_widths = [d['weight'] * 0.5 for _, _, d in T.edges(data=True)]

    nx.draw(T, pos, with_labels=True, node_size=node_sizes, width=edge_widths, font_size=9, node_color='lightgreen', edge_color='grey')
    plt.title("Topic Co-occurrence Network (Connected by Shared Guests)", size=20)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "topic_cooccurrence_network.png", dpi=300)
    plt.show()
    print(f"Topic co-occurrence network saved to {DATA_DIR / 'topic_cooccurrence_network.png'}")
    nx.write_gexf(T, DATA_DIR / "topic_cooccurrence_network.gexf")
    export_interactive_network(
        T,
        DATA_DIR / "topic_cooccurrence_network.html",
        value_attr="num_guests",
        tooltip_labels={"num_guests": "Einzigartige Gäste"},
    )


def _auto_device() -> str:
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _get_embedder(model_name: str, device: Optional[str] = None):
    """
    Load SentenceTransformer with memory-safe defaults on Apple MPS:
    - fp16 on MPS
    - reduced max_seq_length (256)
    Falls back to CPU on OOM.
    """
    import os
    import torch
    from sentence_transformers import SentenceTransformer

    # Optional: relax MPS high-watermark (use with caution)
    # os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    device = device or _auto_device()

    def _load(dev: str, dtype=None):
        kwargs = {}
        if dtype is not None:
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        model = SentenceTransformer(model_name, device=dev, **kwargs)
        # Keep sequences shorter for teasers; reduces memory a lot
        try:
            model.max_seq_length = min(getattr(model, "max_seq_length", 512), 256)
        except Exception:
            pass
        return model

    try:
        if device == "mps":
            # Try half precision first on MPS
            model = _load("mps", dtype=torch.float16)
        elif device == "cuda":
            # Prefer bf16 if supported, else fp16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = _load("cuda", dtype=dtype)
        else:
            model = _load("cpu", dtype=None)
        return model, device
    except RuntimeError as e:
        # Fallback ladder on OOM or unsupported dtype
        msg = str(e).lower()
        if "out of memory" in msg or "mps" in msg:
            try:
                # Try CPU as last resort
                model = _load("cpu", dtype=None)
                return model, "cpu"
            except Exception as e2:
                raise e2
        raise

def _embed_texts(embedder, texts: List[str], batch_size: int = 64):
    """
    Encode with retries:
    - progressively smaller batch sizes on OOM
    - final fallback to CPU if we started on MPS/CUDA
    """
    import numpy as np
    import torch

    def _try_encode(bs: int):
        return embedder.encode(
            texts,
            batch_size=bs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    # First attempt with requested batch size
    try:
        return _try_encode(batch_size)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise

    # Retry with smaller batches
    for bs in [32, 16, 8, 4, 2, 1]:
        try:
            return _try_encode(bs)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

    # Final fallback: move model to CPU and try again
    try:
        embedder.to("cpu")
        return _try_encode(32)
    except Exception:
        # Last resort small batch
        return _try_encode(4)


############################################
# 1) Improved hyperparameter tuning
############################################

def tune_bertopic_hyperparameters(
    df: pd.DataFrame,
    text_col: str = "description",
    min_chars: int = 40,
    min_words: int = 8,
    model_name: str = "intfloat/multilingual-e5-base",
    save_results: bool = True,
    results_path: str | Path = "bertopic_tuning_results.csv",
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Systematically tunes BERTopic hyperparameters and returns the best configuration.

    Args:
        df: DataFrame with a text column.
        text_col: Name of the text column.
        min_chars: Minimum char length to keep a document.
        min_words: Minimum token count to keep a document.
        model_name: SentenceTransformer model to use.
        save_results: Whether to save a CSV of all runs.
        results_path: Where to save the CSV.
        random_state: Random seed for UMAP/HDBSCAN.

    Returns:
        dict: Best parameters and their scores, or None.
    """
    print("\n--- Tuning BERTopic Hyperparameters ---")
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        return None

    if text_col not in df.columns:
        print(f"Column '{text_col}' not found in df.")
        return None

    # Filter & de-duplicate (exact duplicates)
    docs_df = (
        df[df[text_col].notna()]
        .assign(_len=df[text_col].fillna("").str.len(),
                _wc=df[text_col].fillna("").str.split().str.len())
    )
    docs_df = docs_df[(docs_df["_len"] >= min_chars) & (docs_df["_wc"] >= min_words)].copy()
    docs_df.drop_duplicates(subset=[text_col], inplace=True)

    if len(docs_df) < 100:
        print(f"Only {len(docs_df)} docs after filtering. Need at least 100 for reliable tuning.")
        return None

    descriptions = docs_df[text_col].tolist()
    print(f"Tuning on {len(descriptions)} descriptions...")

    # Embeddings (single pass, normalized)
    embedder, device = _get_embedder(model_name)
    print(f"Embedding with '{model_name}' on device '{device}'...")
    E = _embed_texts(embedder, descriptions, batch_size=64)

    # Parameter grid (kept compact & impactful)
    # Tip: n_components fixed low (5) keeps HDBSCAN stable & fast.
    param_grid = {
        "min_cluster_size": [20, 30, 40, 50],
        "min_samples": [5, 10, 15],
        "n_neighbors": [10, 15, 25, 35],
        "min_dist": [0.0, 0.1, 0.25],
        "n_components": [5],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(combos)
    print(f"Testing {total} parameter combinations...")

    results: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Optional[Dict[str, Any]] = None

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        print(f"\n[{i}/{total}] Testing: {params}")

        try:
            # Build UMAP/HDBSCAN as in BERTopic
            umap_model = UMAP(
                n_neighbors=params["n_neighbors"],
                n_components=params["n_components"],
                min_dist=params["min_dist"],
                metric="cosine",
                random_state=random_state,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
                metric="euclidean",  # on UMAP space
                cluster_selection_method="eom",
                prediction_data=True,
            )

            # IMPORTANT: to score consistently, we compute UMAP once here and silhouette on UMAP coords
            E_umap = umap_model.fit_transform(E)

            # Fit HDBSCAN on UMAP space to get labels like BERTopic
            labels = hdbscan_model.fit_predict(E_umap)

            # Compute topic count (exclude -1)
            unique = set(labels)
            num_topics = len(unique - {-1})
            outlier_ratio = float(np.mean(labels == -1))

            # Silhouette: only if we have at least 2 clusters and >= 2 non-outliers
            if num_topics >= 2 and np.sum(labels != -1) >= 2:
                sil = silhouette_score(E_umap[labels != -1], labels[labels != -1], metric="euclidean")
            else:
                sil = -1.0

            # Balanced composite score
            # - prefer lower outliers
            # - prefer 20–60 topics for 3–5k docs, but allow flexibility
            target_lo, target_hi = 20, 60
            if num_topics <= 0:
                topic_term = 0.0
            elif num_topics < target_lo:
                topic_term = num_topics / target_lo  # penalize too few
            elif num_topics > target_hi:
                topic_term = max(0.0, 1 - (num_topics - target_hi) / (2 * target_hi))  # soft penalty
            else:
                topic_term = 1.0

            score = (
                0.45 * sil +
                0.35 * (1 - outlier_ratio) +
                0.20 * topic_term
            )

            row = dict(params,
                       num_topics=int(num_topics),
                       outlier_ratio=float(outlier_ratio),
                       silhouette_score=float(sil),
                       composite_score=float(score))
            results.append(row)

            print(f" -> topics={num_topics} | outliers={outlier_ratio:.2%} | silhouette={sil:.3f} | score={score:.3f}")

            if score > best_score:
                best_score = score
                best_params = row.copy()

        except Exception as e:
            print(f" -> Error in parameter combination {params}: {type(e).__name__}: {e}")

    if not results:
        print("No successful runs.")
        return None

    res_df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    if save_results:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to '{results_path}'")

    # Top 5 preview
    print("\n--- Top 5 ---")
    for _, r in res_df.head(5).iterrows():
        print(f"score={r.composite_score:.3f} | topics={r.num_topics} | outliers={r.outlier_ratio:.2%} | sil={r.silhouette_score:.3f} | "
              f"min_cluster={r.min_cluster_size} min_samples={r.min_samples} n_neighbors={r.n_neighbors} min_dist={r.min_dist} n_comp={r.n_components}")

    if best_params:
        best_params["document_count"] = len(descriptions)
        print(f"\nBest params (score {best_params['composite_score']:.3f}): { {k: best_params[k] for k in keys} }")
        return best_params

    return None


############################################
# 2) Improved analysis + visualization
############################################

def analyze_and_visualize_topics(
    df,
    best_params=None,
    text_col: str = "description",
    uid_col: str | None = "uid",
    date_col: str | None = "date",          # expects '%d.%m.%Y' but auto-parsing is used
    data_dir: str | Path = "data",
    model_name: str = "intfloat/multilingual-e5-base",
    min_words: int = 8,
    save_excel: bool = True,
):
    """
    One-stop BERTopic pipeline with:
      - robust German cleaning (lemmatization, POS filter, custom stopwords)
      - stable modeling on Apple Silicon/CPU
      - improved topic labels via cleaned docs
      - DYNAMIC TOPIC MODELING (manual, bullet-proof; no Interval.left errors)
      - HTML visualizations + Excel export

    Returns:
      DataFrame with topic + topic_label merged back.
    """

    # Base deps
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing BERTopic stack. Install:\n  pip install bertopic sentence-transformers umap-learn hdbscan")
        return df

    # Extra reps (optional)
    try:
        from bertopic.vectorizers import ClassTfidfTransformer
    except Exception:
        ClassTfidfTransformer = None
    try:
        from bertopic.representation import KeyBERTInspired
    except Exception:
        KeyBERTInspired = None

    # _auto_device, _get_embedder, _embed_texts sind auf Modulebene definiert
    # und werden hier direkt genutzt (keine Duplikate nötig).

    # ----------------------- data prep -----------------------
    if text_col not in df.columns:
        print(f"Column '{text_col}' not found.")
        return df

    docs_df = df[df[text_col].notna()].copy()
    docs_df["_wc"] = docs_df[text_col].str.split().str.len()
    docs_df = docs_df[docs_df["_wc"] >= min_words].copy()
    docs_df.drop(columns=["_wc"], inplace=True)

    if len(docs_df) < 20:
        print("Not enough documents (need >= 20).")
        return df

    texts = docs_df[text_col].astype(str).tolist()
    print(f"--- Analyzing Topics ---\nAnalyzing {len(texts)} documents...")

    # ----------------------- modeling -----------------------
    embedder, device = _get_embedder(model_name)
    print(f"Embedding on device: {device}")
    E = _embed_texts(embedder, texts, batch_size=64)

    stopwords = get_german_stopwords()
    vectorizer = CountVectorizer(stop_words=list(stopwords), min_df=2, max_df=0.6, ngram_range=(1, 3))

    if best_params is None:
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=35, min_samples=10, metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True)
    else:
        umap_model = UMAP(n_neighbors=int(best_params["n_neighbors"]),
                          n_components=int(best_params["n_components"]),
                          min_dist=float(best_params["min_dist"]),
                          metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=int(best_params["min_cluster_size"]),
                                min_samples=int(best_params["min_samples"]),
                                metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="german",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, E)

    # Get the number of topics, excluding the outlier topic (-1)
    # The `get_n_topics` method is not available in older BERTopic versions.
    # This is a compatible way to calculate the number of topics, excluding outliers.
    unique_topics = set(topics)
    n_topics = len(unique_topics) - 1 if -1 in unique_topics else len(unique_topics)

    # ----------------------- better labels via cleaned docs -----------------------
    cleaned_texts = [clean_description_for_labels(t) for t in texts]

    try:
        # 1) Recompute topic words using cleaned docs (ask for more; we'll prune to 10)
        vectorizer_for_labels = CountVectorizer(
            stop_words=None, ngram_range=(1, 3), min_df=2, max_df=0.6
        )
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True) if ClassTfidfTransformer is not None else None

        topic_model.update_topics(
            cleaned_texts,
            vectorizer_model=vectorizer_for_labels,
            ctfidf_model=ctfidf if ctfidf is not None else None,
            top_n_words=25
        )

        # 2) Version-safe KeyBERTInspired: use update_topics(..., representation_model=rep)
        if KeyBERTInspired is not None:
            try:
                # Newer signature may accept args; if not, fall back to bare init
                try:
                    rep = KeyBERTInspired(diversity=0.7, nr_candidates=50, random_state=42)
                except TypeError:
                    rep = KeyBERTInspired()
                topic_model.update_topics(
                    cleaned_texts,
                    representation_model=rep,   # <-- key: pass through update_topics
                    top_n_words=25
                )
            except Exception as e:
                print("KeyBERTInspired refinement skipped:", e)

        # --- de-duplicate shorter n-grams contained in longer phrases ---
        topics_dict = topic_model.get_topics()
        deduped = _dedupe_topic_terms(topics_dict, keep_n=10)
        filtered_repr = _filter_topic_terms(deduped)

        # Write back deduped representations if supported; otherwise we'll just set labels
        wrote_reprs = False
        if hasattr(topic_model, "set_topic_representations"):
            try:
                topic_model.set_topic_representations(filtered_repr)
                wrote_reprs = True
            except Exception as e:
                print(f"Warning: Could not set topic representations: {type(e).__name__}: {e}")

        # Build clean labels: prefer the longest phrase available
        labels = {}
        for tid, terms in filtered_repr.items():
            if tid == -1 or not terms:
                continue
            best_phrase = max((t for t, _ in terms), key=lambda w: len(w.split()), default=terms[0][0])
            labels[tid] = best_phrase  # keep original casing; or use .title() if you prefer

        try:
            topic_model.set_topic_labels(labels)
        except Exception as e:
            print(f"Warning: Could not set topic labels: {type(e).__name__}: {e}")

    except Exception as e:
        print("Label refinement skipped:", e)

    # ----------------------- map labels back -----------------------
    topic_info = topic_model.get_topic_info()
    # if set_topic_labels worked, Name already reflects our labels; still normalize spacing a bit
    topic_info["custom_label"] = (
        topic_info["Name"]
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    label_map = topic_info.set_index("Topic")["custom_label"].to_dict()

    docs_df["topic"] = topics
    docs_df["topic_label"] = docs_df["topic"].map(label_map)

    # ----------------------- save model & static visuals -----------------------
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_path = data_dir / "talkshow_topic_model"
    topic_model.save(str(model_path), serialization="safetensors")
    print(f"Topic model saved to {model_path}")
    # Überschreibe die von topic_model.save() gespeicherten Repräsentationen mit
    # unseren bereinigten (Rollenfilter + Dedupe), damit topics.json die finalen
    # Labels enthält statt der rohen KeyBERTInspired-Ausgabe.
    try:
        saved_topics_path = model_path / "topics.json"
        if saved_topics_path.exists():
            import json as _json
            with open(saved_topics_path, encoding="utf-8") as _f:
                _saved = _json.load(_f)
            _saved["topic_representations"] = {
                str(tid): [(w, float(s)) for w, s in terms]
                for tid, terms in filtered_repr.items()
            }
            with open(saved_topics_path, "w", encoding="utf-8") as _f:
                _json.dump(_saved, _f, ensure_ascii=False, indent=2)
            print("topics.json mit bereinigten Repräsentationen (Rollenfilter + Dedupe) überschrieben.")
    except Exception as _e:
        print(f"Warning: Konnte topics.json nicht nachträglich aktualisieren: {_e}")

    out_df = df.copy()
    if uid_col and uid_col in df.columns and uid_col in docs_df.columns:
        out_df = out_df.merge(docs_df[[uid_col, "topic", "topic_label"]], on=uid_col, how="left")
    else:
        out_df = out_df.join(docs_df[["topic", "topic_label"]])

    if save_excel:
        xlsx_path = data_dir / "all_data_with_topics.xlsx"
        out_df.to_excel(xlsx_path, index=False)
        print(f"Data with topics saved to {xlsx_path}")

    print("Generating visualizations...")
    try:
        topic_model.visualize_topics().write_html(str(data_dir / "topics_visualization.html"))
        topic_model.visualize_hierarchy(top_n_topics=n_topics).write_html(str(data_dir / "topics_hierarchy.html"))
        topic_model.visualize_barchart(top_n_topics=n_topics).write_html(str(data_dir / "topics_barchart.html"))
        print("Visualizations written to:", data_dir)
    except Exception as e:
        print("Static visualization error:", e)

    # ----------------------- bullet-proof DYNAMICS (manual) -----------------------
    try:
        if date_col and date_col in docs_df.columns:
            ts_series = pd.to_datetime(docs_df[date_col], errors="coerce")
            pairs = [(t, ts) for t, ts in zip(texts, ts_series)
                     if isinstance(t, str) and t.strip() != "" and pd.notna(ts)]
            if not pairs:
                print("No valid (doc, timestamp) pairs; skipping dynamics.")
            else:
                texts_dyn, ts_dyn = zip(*pairs)
                texts_dyn = list(texts_dyn)
                ts_dyn = pd.Series(ts_dyn).astype("datetime64[ns]")

                # Re-assign topics for these docs with the current model
                topics_dyn, _ = topic_model.transform(texts_dyn)

                # Drop outliers if not desired in chart
                keep_mask = np.array(topics_dyn) != -1
                topics_dyn = np.array(topics_dyn)[keep_mask]
                ts_dyn = ts_dyn[keep_mask].reset_index(drop=True)

                uniq_points = int(ts_dyn.nunique())
                uniq_years = int(ts_dyn.dt.year.nunique())
                if len(ts_dyn) < 2 or uniq_points < 2:
                    print("Too few unique timestamps for dynamics; skipping.")
                else:
                    base_bins = max(10, min(40, uniq_years))
                    nr_bins = max(2, min(base_bins, uniq_points))

                    print(f"Creating dynamics with nr_bins={nr_bins} "
                          f"(unique years={uniq_years}, unique timestamps={uniq_points}, "
                          f"docs with dates={len(ts_dyn)})")

                    tmin, tmax = ts_dyn.min(), ts_dyn.max()
                    if tmax <= tmin:
                        print("Degenerate time span; skipping dynamics.")
                    else:
                        edges = pd.date_range(start=tmin, end=tmax, periods=nr_bins + 1)
                        labels = pd.cut(ts_dyn, bins=edges, labels=False, include_lowest=True, right=False)

                        ok = pd.notna(labels)
                        ts_dyn = ts_dyn[ok].reset_index(drop=True)
                        topics_dyn = np.array(topics_dyn)[ok.values]
                        labels = labels[ok].astype(int)

                        mids = edges[:-1] + (edges[1:] - edges[:-1]) / 2
                        timepoints = pd.Series(mids, name="Timestamp")

                        df_dyn = pd.DataFrame({"Topic": topics_dyn, "Bin": labels})
                        freq = (df_dyn.groupby(["Topic", "Bin"], as_index=False)
                                      .size()
                                      .rename(columns={"size": "Frequency"}))
                        freq["Timestamp"] = freq["Bin"].map(dict(enumerate(timepoints)))
                        freq.drop(columns=["Bin"], inplace=True)

                        def topic_words(t, top_n=10):
                            kws = topic_model.get_topic(int(t)) or []
                            return " | ".join(w for w, _ in kws[:top_n]) if kws else ""
                        freq["Words"] = freq["Topic"].apply(topic_words)

                        tot = freq[["Topic", "Timestamp", "Words", "Frequency"]]

                        topic_model.visualize_topics_over_time(
                            tot, top_n_topics=15, normalize_frequency=True
                        ).write_html(str(data_dir / "topics_over_time.html"))
        else:
            print("No valid date column for dynamics; skipping.")
    except Exception as e:
        print("Dynamic topic modeling failed:", e)

    return out_df


############################################
# 3) Main pipeline helper functions
############################################

def load_all_show_data() -> tuple[list[dict], pd.DataFrame]:
    """
    Load all show data from JSON files and prepare initial DataFrame.

    Returns:
        tuple: (all_data, df) where all_data is the raw list of dicts and df is the DataFrame
    """
    show_files = [
        "AnneWill_data.json", "CarenMiosga_data.json", "HartAberFair_data.json",
        "MarkusLanz_data.json", "Maischberger_data.json", "Illner_data.json"
    ]

    all_data = []
    for file_name in show_files:
        all_data.extend(load_json_file(DATA_DIR / file_name))

    all_data = [dat for dat in all_data if dat is not None]

    df = pd.DataFrame(all_data)
    df['description'] = df['description'].fillna('')  # Ensure no NaN in description

    return all_data, df


def get_or_tune_parameters(df: pd.DataFrame, model_name: str = "intfloat/multilingual-e5-base") -> Optional[Dict[str, Any]]:
    """
    Load existing BERTopic parameters or run hyperparameter tuning if needed.

    Args:
        df: DataFrame with description column
        model_name: SentenceTransformer model name

    Returns:
        dict: Best parameters, or None if tuning fails
    """
    best_params = load_json_file(DATA_DIR / "best_parameters.json")
    descriptions = df['description'].tolist()

    # Retune if params don't exist or document count has grown significantly
    if not best_params or best_params.get("document_count", 0) < 0.8 * len(descriptions):
        print("\nNo valid best parameters found or insufficient document count. Starting hyperparameter tuning...")
        best_params = tune_bertopic_hyperparameters(df, model_name=model_name)

    if best_params:
        save_json_file(DATA_DIR / "best_parameters.json", best_params)
        print(f"Successfully saved best parameters to {DATA_DIR / 'best_parameters.json'}")

    return best_params


def perform_guest_analysis(
    all_data: list[dict],
    df_with_topics: pd.DataFrame,
    model_name: str = "intfloat/multilingual-e5-base",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform guest analysis: clean, consolidate, categorize, and merge with topics.

    Args:
        all_data:     Raw episode data with guests
        df_with_topics: DataFrame with topic assignments
        model_name:   SentenceTransformer model for embedding-based fallback

    Returns:
        tuple: (df_guests_with_topics, df_cleaned, df_consolidated)
    """
    print("\n--- Guest Analysis ---")

    # Prepare and clean guest data
    df_guests = prepare_guest_dataframe(all_data)
    df_cleaned, df_junk = clean_guest_rows(
        df_guests,
        name_col="name",
        role_col="role",
        party_col="party",
        show_col="Talkshow"
    )

    # Consolidate to one row per unique guest
    df_consolidated = consolidate_guests(df_cleaned)

    # Rename the 'name_clean' column to 'name' for consistency
    if 'name_clean' in df_consolidated.columns:
        df_consolidated.rename(columns={'name_clean': 'name'}, inplace=True)

    # Schritt 1: Regelbasierte Klassifizierung
    df_categorized = add_classification_columns(df_consolidated)

    # Schritt 2: Embedding-Fallback für Gäste ohne Regelkategorie
    try:
        embedder, _ = _get_embedder(model_name)
        embed_fn = lambda texts: _embed_texts(embedder, texts)
        prototype_embeddings = build_prototype_embeddings(embed_fn)
        df_categorized = apply_embedding_fallback(
            df_categorized,
            embed_fn,
            prototype_embeddings,
            role_col="role_clean",
            desc_col="description",
        )
    except Exception as e:
        print(f"[embedding-fallback] Nicht verfügbar, überspringe: {e}")

    # Save guest data to Excel files
    df_categorized.to_excel(DATA_DIR / 'guests_consolidated.xlsx', index=False)
    print(f"Consolidated guest list saved to {DATA_DIR / 'guests_consolidated.xlsx'}")

    df_cleaned.to_excel(DATA_DIR / 'guests_cleaned.xlsx', index=False)
    print(f"Cleaned guest list saved to {DATA_DIR / 'guests_cleaned.xlsx'}")

    df_junk.to_excel(DATA_DIR / 'guests_junk.xlsx', index=False)
    print(f"Junk guest list saved to {DATA_DIR / 'guests_junk.xlsx'}")

    # Merge guest data with topic data for integrated analysis
    df_guests_with_topics = pd.merge(
        df_cleaned,
        df_with_topics[['uid', 'topic', 'topic_label']],
        on='uid',
        how='left'
    )

    # Rename the 'name_clean' column to 'name' for consistency in downstream analysis
    if 'name_clean' in df_guests_with_topics.columns:
        if 'name' in df_guests_with_topics.columns:
            df_guests_with_topics.drop(columns=['name'], inplace=True)
        df_guests_with_topics.rename(columns={'name_clean': 'name'}, inplace=True)

    df_guests_with_topics.to_excel(DATA_DIR / "guests_with_topics.xlsx", index=False)
    print(f"Detailed guest list with topics saved to {DATA_DIR / 'guests_with_topics.xlsx'}")

    return df_guests_with_topics, df_cleaned, df_categorized


def create_network_visualizations(df_guests_with_topics: pd.DataFrame) -> None:
    """
    Create and save guest co-occurrence network visualizations.

    Args:
        df_guests_with_topics: DataFrame with guest and topic information
    """
    print("\n--- Creating Network Visualizations ---")

    # Analyze guest-topic connections (creates topic co-occurrence network)
    analyze_topic_guest_connections(df_guests_with_topics)

    # Create guest co-occurrence network
    grouped = df_guests_with_topics.groupby('Talkshow')['name'].apply(list)
    edges = [
        edge for names in grouped
        for edge in itertools.combinations(sorted(list(set(names))), 2)
    ]
    edge_counts = Counter(edges)

    G = nx.Graph()
    for edge, weight in edge_counts.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    # Add topic range as a node attribute to the guest network
    valid_topics_df = df_guests_with_topics.dropna(subset=['topic_label'])
    valid_topics_df = valid_topics_df[valid_topics_df['topic'] != -1]
    guest_topic_range = valid_topics_df.groupby('name')['topic_label'].nunique()
    nx.set_node_attributes(G, guest_topic_range.to_dict(), 'topic_range')

    appearance_attr = df_guests_with_topics['name'].value_counts().to_dict()
    nx.set_node_attributes(G, appearance_attr, 'appearances')

    # Visualize and export
    visualize_network(
        G,
        df_guests_with_topics,
        output_path=DATA_DIR / "cooccurrence_network.png",
    )

    nx.write_gexf(G, DATA_DIR / "cooccurrence_network_with_weights.gexf")
    export_interactive_network(
        G,
        DATA_DIR / "cooccurrence_network.html",
        value_attr='appearances',
        tooltip_labels={'appearances': 'Auftritte', 'topic_range': 'Themenvielfalt'},
        min_edge_weight=4,
        max_nodes=120,
        label_top_n=24,
        community_colors=True,
    )


def main():
    """Main function to run the talkshow analysis pipeline."""
    # 1. Load all show data
    all_data, df = load_all_show_data()

    # 2. Get or tune BERTopic parameters
    best_params = get_or_tune_parameters(df)

    # 3. Perform topic modeling
    if best_params:
        print("\nUsing best parameters from tuning for topic modeling.")
        df_with_topics = analyze_and_visualize_topics(df, best_params=best_params)
    else:
        df_with_topics = analyze_and_visualize_topics(df)

    # 4. Perform guest analysis (clean, consolidate, categorize, merge with topics)
    df_guests_with_topics, df_cleaned, df_consolidated = perform_guest_analysis(
        all_data, df_with_topics
    )

    # 5. Create network visualizations
    create_network_visualizations(df_guests_with_topics)

if __name__ == "__main__":
    main()
