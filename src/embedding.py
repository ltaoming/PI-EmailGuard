from sentence_transformers import SentenceTransformer
from pathlib import Path
from datasets import load_from_disk
import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm

# json_path = (Path(__file__).parent.parent / "data" / "merged_email_dataset.json").resolve()
# if not json_path.exists():
#     raise FileNotFoundError(f"JSON file not found: {json_path}")
# with json_path.open("r", encoding="utf-8") as f:
#     data = json.load(f)
# df = pd.read_json("../data/merged_email_dataset.json")
df = pd.read_json("hf://datasets/0tt00t/PI-EmailGuard/merged_email_dataset.json")

output_column = df["output"].tolist()

sentences = output_column

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
start_time = time.time()
embeddings = []
embeddings = model.encode(sentences)
end_time = time.time()

print(f"Time taken to generate embeddings: {end_time - start_time:.2f} seconds")
# print(embeddings)

