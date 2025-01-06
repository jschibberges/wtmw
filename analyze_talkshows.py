import os
import json
import pandas as pd
from datetime import datetime, timedelta
import locale
import hashlib
import base64
import re
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from collections import Counter

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
        with open(filepath, 'r', encoding='utf-8') as f:  # Explicitly handle encoding
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return None
    except Exception as e:  # Catch other potential errors
        print(f"An unexpected error occurred: {e}")
        return None

def clean_str(text):
    if text is not None:
        text = text.strip()
        if "| Bild: " in text:
            text = ""
    else:
        text=""
    return text

all_annewill_data = load_json_file("./data/AnneWill_data.json")
all_carenmiosga_data = load_json_file("./data/CarenMiosga_data.json")
all_hartaberfair_data = load_json_file("./data/HartAberFair_data.json")
all_markuslanz_data = load_json_file("./data/MarkusLanz_data.json")
all_maischberger_data = load_json_file("./data/Maischberger_data.json")
all_illner_data = load_json_file("./data/Illner_data.json")

all_data = all_annewill_data+all_carenmiosga_data+all_hartaberfair_data+all_markuslanz_data+all_maischberger_data+all_illner_data
df = pd.DataFrame(all_data)


# Aggregate
#df.to_excel("all_data.xlsx", index=False)
guest_list = []
for dat in all_data:
    if "guests" in dat and dat["guests"] is not None:
        for guest in dat["guests"]:
            dic = {}
            dic.update(guest)
            dic["Talkshow"] = dat["show"]+" - "+str(dat["date"])
            guest_list.append(dic)
            if len(guest["name"])>37:
                print(dat["link"])
                print(guest["name"])

def manual_name_corrections(name):
    if name == "Annabel Oelsmann":
        name = "Annabel Oelmann"
    elif name == "Kateryna Mishenko":
        name = "Kateryna Mishchenko"
    elif name == "Marie-A. Strack-Zimmermann" or name == "Strack-Zimmermann":
        name = "Marie-Agnes Strack-Zimmermann"
    elif name == "Melanie Amman":
        name = "Melanie Amann"
    elif name == 'Christiane Hoffmann Autorin Hauptstadtbüro "Der Spiegel"':
        name = "Christiane Hoffmann"
    elif name == "Hermann Josef Tenhagen":
        name = "Hermann-Josef Tenhagen"
    elif name == "Ranga Yogeshwar Wissenschaftsjournalist":
        name = "Ranga Yogeshwar"
    elif name == "Amira Mohammed Ali":
        name = "Amira Mohamed Ali"

    return name

df_guests = pd.DataFrame(guest_list)
df_guests['name'] = df_guests['name'].apply(manual_name_corrections)
df_guests['role'] = df_guests['role'].apply(clean_str)
df_guests['description'] = df_guests['description'].apply(clean_str)
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

df_consolidated = consolidate_guests(df_guests)
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
df_consolidated = remove_party_from_other_columns(df_consolidated)

filename = f'{datetime.now().strftime("%Y%m%d")}_guests.xlsx'
df_guests.to_excel(filename)


grouped = df_guests.groupby('Talkshow')['name'].apply(list)

# Step 2: Generate edges
edges = []
for names in grouped:
    edges.extend(itertools.combinations(names, 2))  # Generate all pairs of people in the same Talkshow

# Count the occurrences of each pair
edge_counts = Counter(edges)

# Step 3: Create the graph with weighted edges
G = nx.Graph()
for edge, weight in edge_counts.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Step 4: Visualize the network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Position nodes using the spring layout

# Draw nodes and edges
nx.draw(
    G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=15
)

# Add edge labels (weights)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

plt.title("Co-occurrence Network of People in Talkshows (with Weights)")
plt.show()

# Step 5: Export the graph (optional)
nx.write_gexf(G, "cooccurrence_network_with_weights.gexf")



'''
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


descriptions = list(df.description)
descriptions = [des for des in descriptions if len(des)>200]

# Pre-calculate embeddings
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedding_model.encode(descriptions, show_progress_bar=True)
# Fine-tune your topic representations
representation_model = KeyBERTInspired()
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine')

topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(descriptions, embeddings)

topic_model.get_topic_info()

# Fine-tune topic representations after training BERTopic
vectorizer_model = CountVectorizer(stop_words=["der", "die","das","in","wie","und","zu",], ngram_range=(1, 3), min_df=2)
topic_model.update_topics(descriptions, vectorizer_model=vectorizer_model)


'''