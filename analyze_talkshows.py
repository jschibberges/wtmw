
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

from scrape_talkshows import load_json_file, save_json_file, LiteConfig
from nlp_utils import (
    clean_guest_rows,
    clean_description_for_labels,
    _dedupe_topic_terms,
    sanitize_person_name,
    clean_person_name,
    get_german_stopwords,
)



# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load data
def load_json_file(filepath):
    """Loads data from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        The loaded JSON data as a Python dictionary or list, or None if 
        an error occurs during loading.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return []
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return []

def clean_str(text):
    if text is not None:
        text = text.strip()
        if "| Bild: " in text:
            text = ""
    else:
        text=""
    return text

def manual_name_corrections(name):
    """Applies manual name corrections from a mapping file."""
    corrections_path = DATA_DIR / "name_corrections.json"
    if not corrections_path.exists():
        print("Warning: name_corrections.json not found. Skipping manual corrections.")
        return name

    name_corrections = load_json_file(corrections_path)
    # Ensure that the loaded data is a dictionary before using .get()
    if isinstance(name_corrections, dict):
        return name_corrections.get(name, name)
    
    # If not a dict (e.g., file was empty or malformed), return original name
    return name

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

    df_guests['name_raw'] = df_guests['name'].apply(lambda x: x)
    df_guests['name'] = df_guests['name'].apply(manual_name_corrections)
    df_guests['name'] = df_guests['name'].apply(lambda x: sanitize_person_name(x) or clean_person_name(x) or None)
    df_guests.dropna(subset=['name'], inplace=True)
    df_guests.drop_duplicates(subset=['name', 'uid'], inplace=True)
    df_guests['role'] = df_guests['role'].apply(clean_str)
    df_guests['description'] = df_guests['description'].apply(clean_str)
    return df_guests

def consolidate_guests(df_guests):
    def consolidate_party(parties):
        if len(parties) == 0:
            return None
        elif len(set(parties)) == 1:
            return parties.iloc[0]
        elif "BSW" in parties:
            return "BSW"
        elif "parteilos" in parties:
            return "parteilos"
        else:
            return parties.mode()[0]  # Return the most frequent

    def consolidate_column(series):
        return list(set(series)) if len(set(series)) > 1 else series.iloc[0]

    # Group by name and aggregate other columns
    df_consolidated = df_guests.groupby("name").agg({
        "party": consolidate_party,  # Custom aggregation for party
        "role": consolidate_column,
        "description": consolidate_column,
        "Talkshow": consolidate_column,
    }).reset_index()

    return df_consolidated

def remove_party_from_other_columns(df):
    """Removes party values from 'role' and 'description' columns if they exist as list elements."""
    def remove_party(row):
        if isinstance(row['role'], list) and row['party'] in row['role']:
            row['role'].remove(row['party'])

        if isinstance(row['description'], list) and row['party'] in row['description']:
            row['description'].remove(row['party'])
        return row

    df = df.apply(remove_party, axis=1)  # Apply function row-wise
    return df

def visualize_network(G, df_guests=None, layout="spring", seed=42, figsize=(12, 9)):
    """
    Draw a co-occurrence graph with nodes colored by the 'topic_range' node attribute.
    Adds a colorbar safely (on the same Axes) when topic_range exists and varies.

    Args:
        G: networkx.Graph with optional node attr 'topic_range'
        df_guests: (optional) DataFrame for tooltips/labels (not required here)
        layout: 'spring' | 'kamada_kawai' | 'fr' | 'spectral' | 'circular'
        seed: random seed for layouts that support it
        figsize: figure size
    """
    # ---- positions
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=None)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(G, seed=seed)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # ---- node sizes (optional): degree-based
    deg = dict(G.degree())
    node_sizes = [300 + 30 * deg.get(n, 0) for n in G.nodes()]

    # ---- node colors from 'topic_range'
    tr_vals = [G.nodes[n].get("topic_range", None) for n in G.nodes()]
    has_tr = any(v is not None for v in tr_vals)

    # default gray if no topic_range
    node_color = "#A0A0A0"
    cmap = mpl.cm.viridis
    norm = None
    color_array = None

    if has_tr:
        # Replace None with np.nan to compute min/max safely
        arr = np.array([np.nan if v is None else float(v) for v in tr_vals], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size >= 1:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if vmin == vmax:
                # Constant color map: avoid singular norm by widening range a bit
                vmin, vmax = vmin - 0.5, vmax + 0.5
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            color_array = arr
        else:
            has_tr = False  # all NaN -> fall back to default gray

    # ---- draw
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # edges: light gray
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25)

    # nodes
    if has_tr and color_array is not None:
        # Map NaNs (if any) to a neutral color by replacing with midpoint
        midpoint = (norm.vmin + norm.vmax) / 2.0
        color_array = np.where(np.isfinite(color_array), color_array, midpoint)
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=color_array, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax,
            node_size=node_sizes, linewidths=0.5, edgecolors="white"
        )
    else:
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_color, node_size=node_sizes,
            linewidths=0.5, edgecolors="white"
        )

    # labels (optional; comment out if cluttered)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    # ---- colorbar: only if we actually colored by topic_range
    if has_tr and norm is not None:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        # set_array is needed in older Matplotlib to enable colorbar scale
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Guest Topic Range")

    fig.tight_layout()
    plt.show()

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
            print(f" -> Error: {e}")

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

    # NLP cleaning deps
    try:
        import spacy
        nlp_de = spacy.load("de_core_news_md", disable=["parser", "ner"])
    except Exception:
        try:
            import spacy
            nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])
            print("Using spaCy 'sm' model. For better lemmatization run: python -m spacy download de_core_news_md")
        except Exception:
            nlp_de = None
            print("spaCy German model not available; proceeding without lemmatization.")

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
        import torch
        device = device or _auto_device()

        def _load(dev: str, dtype=None):
            kwargs = {}
            if dtype is not None:
                kwargs["model_kwargs"] = {"torch_dtype": dtype}
            model = SentenceTransformer(model_name, device=dev, **kwargs)
            try:
                model.max_seq_length = min(getattr(model, "max_seq_length", 512), 256)
            except Exception:
                pass
            return model

        try:
            if device == "mps":
                return _load("mps", dtype=torch.float16), "mps"
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                return _load("cuda", dtype=dtype), "cuda"
            return _load("cpu", dtype=None), "cpu"
        except RuntimeError:
            return _load("cpu", dtype=None), "cpu"

    def _embed_texts(embedder, texts: List[str], batch_size: int = 64):
        def _try(bs: int):
            return embedder.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
        try:
            return _try(batch_size)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        for bs in [32, 16, 8, 4, 2, 1]:
            try:
                return _try(bs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
        embedder.to("cpu")
        return _try(16)

    def normalize_umlauts(s: str) -> str:
        return unicodedata.normalize("NFKC", s)

    def clean_german_text(text: str,
                          keep_pos: Optional[set] = {"NOUN", "PROPN", "ADJ", "VERB"},
                          min_len: int = 2) -> str:
        """Lemmatize (if spaCy available), keep informative POS, drop stopwords."""
        text = normalize_umlauts(text or "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        stop_words = get_german_stopwords()
        if nlp_de is None:
            # fallback: lowercase + simple token filter
            toks = [t for t in re.findall(r"\b\w+\b", text.lower())
                    if len(t) >= min_len and t not in stop_words]
            return " ".join(toks)

        doc = nlp_de(text)
        toks = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_url or t.like_email or t.like_num:
                continue
            if keep_pos and t.pos_ not in keep_pos:
                continue
            lemma = (t.lemma_ or t.text).lower().strip("._:;,'\"()[]!?-")
            if len(lemma) < min_len:
                continue
            if lemma in stop_words:
                continue
            toks.append(lemma)
        return " ".join(toks)

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
    cleaned_texts = [clean_german_text(t) for t in texts]

    # helper: remove redundant n-grams (keep longer phrases first)
    def _dedupe_topic_terms(topics_dict: dict[int, list[tuple[str, float]]],
                            keep_n: int = 10) -> dict[int, list[tuple[str, float]]]:
        new_repr: dict[int, list[tuple[str, float]]] = {}
        for tid, terms in topics_dict.items():
            if tid == -1 or not terms:
                new_repr[tid] = terms
                continue
            cand = sorted(terms, key=lambda x: (-len(x[0].split()), -x[1]))
            kept, kept_sets = [], []
            for w, s in cand:
                toks = tuple(t for t in w.split() if t)
                wset = set(toks)
                if any(wset <= ks for ks in kept_sets):  # drop if subset of a kept phrase
                    continue
                kept.append((w, s))
                kept_sets.append(wset)
                if len(kept) >= keep_n:
                    break
            new_repr[tid] = kept
        return new_repr

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

        from functools import lru_cache

        LOW_INFORMATION_LEMMAS = {
            "deutsch", "deutschland", "bundesrepublik", "jahr", "jaehrig", "jährig",
            "jähr", "jährlich", "uhr",
        }

        @lru_cache(maxsize=512)
        def _phrase_lemmas(phrase: str) -> tuple[str, ...]:
            phrase = (phrase or "").strip()
            if not phrase:
                return tuple()
            normalized = normalize_umlauts(phrase.lower())
            if nlp_de is not None:
                doc = nlp_de(normalized)
                return tuple(
                    (t.lemma_ or t.text).lower().strip("._:;,'\"()[]!?-")
                    for t in doc if (t.lemma_ or t.text)
                )
            tokens = re.findall(r"\b\w+\b", normalized)
            return tuple(tokens)

        def _filter_topic_terms(terms_by_topic: dict[int, list[tuple[str, float]]]) -> dict[int, list[tuple[str, float]]]:
            filtered: dict[int, list[tuple[str, float]]] = {}
            for tid, terms in terms_by_topic.items():
                if tid == -1 or not terms:
                    filtered[tid] = terms
                    continue
                seen_keys: set[tuple[str, ...]] = set()
                cleaned: list[tuple[str, float]] = []
                for word, score in terms:
                    lemmas = tuple(l for l in _phrase_lemmas(word) if l)
                    if not lemmas:
                        continue
                    if all(l in LOW_INFORMATION_LEMMAS for l in lemmas):
                        continue
                    key = lemmas
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    cleaned.append((word, score))
                if not cleaned:
                    cleaned = terms
                filtered[tid] = cleaned
            return filtered
        
        # --- de-duplicate shorter n-grams contained in longer phrases ---
        topics_dict = topic_model.get_topics()
        deduped = _dedupe_topic_terms(topics_dict, keep_n=10)
        filtered_repr = _filter_topic_terms(deduped)

        # Write back deduped representations if supported; otherwise we’ll just set labels
        wrote_reprs = False
        if hasattr(topic_model, "set_topic_representations"):
            try:
                topic_model.set_topic_representations(filtered_repr)
                wrote_reprs = True
            except Exception as _:
                pass

        # Build clean labels: prefer the longest phrase available
        labels = {}
        for tid, terms in filtered_repr.items():
            if tid == -1 or not terms:
                continue
            best_phrase = max((t for t, _ in terms), key=lambda w: len(w.split()), default=terms[0][0])
            labels[tid] = best_phrase  # keep original casing; or use .title() if you prefer

        try:
            topic_model.set_topic_labels(labels)
        except Exception as _:
            # If setting labels fails on your version, we’ll fall back to Name-based mapping below
            pass

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

def main():
    """Main function to run the analysis."""
    # Load all data
    show_files = [
        "AnneWill_data.json", "CarenMiosga_data.json", "HartAberFair_data.json",
        "MarkusLanz_data.json", "Maischberger_data.json", "Illner_data.json"
    ]

    all_data = []
    for file_name in show_files:
        all_data.extend(load_json_file(DATA_DIR / file_name))

    all_data = [dat for dat in all_data if dat is not None]

    df = pd.DataFrame(all_data)
    df['description'] = df['description'].fillna('') # Ensure no NaN in description
    df_guests = prepare_guest_dataframe(all_data)

    # --- Optional: Hyperparameter Tuning for BERTopic ---
    # This is a long-running process to help find optimal parameters.
    # Run this separately and then update the parameters in analyze_and_visualize_topics.
    best_params = load_json_file(DATA_DIR / "best_parameters.json")
    descriptions = df['description'].tolist()
    if (
        not best_params or
        best_params.get("document_count", 0) < 0.8 * len(descriptions)
    ):
        print("\nNo valid best parameters found or insufficient document count. Starting hyperparameter tuning...")
        best_params = tune_bertopic_hyperparameters(df,model_name="intfloat/multilingual-e5-base")
    if best_params:
        save_json_file(DATA_DIR / "best_parameters.json", best_params)
        print(f"Successfully saved best parameters to {DATA_DIR / "best_parameters.json"}")
    # --- End Optional Tuning ---

    # --- Topic Modeling ---
    if best_params:
        print("\nUsing best parameters from tuning for topic modeling.")
        df_with_topics = analyze_and_visualize_topics(df, best_params=best_params)
    else:
        df_with_topics = analyze_and_visualize_topics(df)
    # --- End Topic Modeling ---

    # --- Guest Analysis ---
    df_guests = prepare_guest_dataframe(all_data)
    #df_consolidated = consolidate_guests(df_guests)
    df_consolidated, df_junk = clean_guest_rows(df_guests, name_col="name", role_col="role", party_col="party", show_col="Talkshow")

    df_consolidated = remove_party_from_other_columns(df_consolidated)

    # Save guest data to Excel
    guest_filename = DATA_DIR / 'consolidated_guests.xlsx'
    df_consolidated.to_excel(guest_filename, index=False)
    print(f"Consolidated guest list saved to {guest_filename}")

    # Merge guest data with topic data for integrated analysis
    df_guests_with_topics = pd.merge(df_guests, df_with_topics[['uid', 'topic', 'topic_label']], on='uid', how='left')
    df_guests_with_topics.to_excel(DATA_DIR / "guests_with_topics.xlsx", index=False)
    print(f"Detailed guest list with topics saved to {DATA_DIR / 'guests_with_topics.xlsx'}")

    # --- New: Analyze Guest-Topic Connections ---
    analyze_topic_guest_connections(df_guests_with_topics)

    # --- Create and visualize the guest co-occurrence network ---
    grouped = df_guests.groupby('Talkshow')['name'].apply(list)
    edges = [edge for names in grouped for edge in itertools.combinations(sorted(list(set(names))), 2)]
    edge_counts = Counter(edges)
    G = nx.Graph()
    for edge, weight in edge_counts.items():
        G.add_edge(edge[0], edge[1], weight=weight)
    
    # Add topic range as a node attribute to the guest network
    valid_topics_df = df_guests_with_topics.dropna(subset=['topic_label'])
    valid_topics_df = valid_topics_df[valid_topics_df['topic'] != -1]
    guest_topic_range = valid_topics_df.groupby('name')['topic_label'].nunique()
    nx.set_node_attributes(G, guest_topic_range.to_dict(), 'topic_range')

    visualize_network(G, df_guests)

    # Export the graph (optional)
    nx.write_gexf(G, DATA_DIR / "cooccurrence_network_with_weights.gexf")

if __name__ == "__main__":
    main()
