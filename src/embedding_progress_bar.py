from sentence_transformers import SentenceTransformer
from pathlib import Path
from datasets import load_from_disk
import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm

# df = pd.read_json("../data/merged_email_dataset.json")
df = pd.read_json("hf://datasets/0tt00t/PI-EmailGuard/merged_email_dataset.json")

output_column = df["output"].tolist()

sentences = output_column

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


batch_size = 1000
n = len(sentences)
emb_list = []
encode_time = 0.0

with tqdm(total=n, desc="Encoding", unit="items") as pbar:
    for i in range(0, n, batch_size):
        batch = sentences[i : i + batch_size]
        t0 = time.time()
        batch_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        t1 = time.time()
        encode_time += (t1 - t0)
        emb_list.append(batch_emb)
        pbar.update(len(batch))

if len(emb_list) > 0:
    embeddings = np.vstack(emb_list)
else:
    embeddings = np.empty((0, model.get_sentence_embedding_dimension()))

print(f"Time taken to generate embeddings (encode only): {encode_time:.2f} seconds")


