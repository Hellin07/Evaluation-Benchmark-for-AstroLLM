# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:40:21 2024

@author: Hailing Lu
"""
import os
from sentence_transformers import SentenceTransformer
import gzip
import json
import torch
import argparse
import time
from tqdm import tqdm
import pyarrow.parquet as pq

def main(args):
    device = torch.device("cpu" if args.gpu is None else f"cuda:{args.gpu}")

    # Load sentence embeddings model
    model = SentenceTransformer(args.model_name, device=device, cache_folder=args.cache_dir)

    # Load astronomy examples text from JSON file
    with open(args.examples_path, "r") as f:
        examples_data = json.load(f)
    examples = [data["text"] for data in examples_data]

    # Embedding examples text
    example_embeddings = model.encode(
        examples,
        show_progress_bar=True,
        batch_size=args.batch_size,
        convert_to_numpy=False,
        normalize_embeddings=True,
        device=device,
    )
    example_embeddings = torch.vstack(example_embeddings)

    # Load documents text from JSONL file
    with open(args.jsonl_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {line}")
                print(f"Error message: {e}")
                raise e
            json_lines.append(data)

    item_count = 0
    for i in tqdm(range(0, len(json_lines), args.batch_size)):
        batch = json_lines[i:i + args.batch_size]
        documents = [data["text"] for data in batch]

        # Embedding documents text
        docs_embeddings = model.encode(
            documents,
            show_progress_bar=False,
            batch_size=args.batch_size,
            convert_to_numpy=False,
            normalize_embeddings=True,
            device=device,
        )
        docs_embeddings = torch.vstack(docs_embeddings)

        # Calculate the similarity matrix of astronomy titles and dolma titles.
        # Columns of the matrix represents dolma titles, and the rows represent astronomy titles.
        title_similar_matrix = torch.matmul(example_embeddings, docs_embeddings.T)
        max_similar = title_similar_matrix.max(dim=0).values

        idx_over_threshold = torch.where(max_similar >= args.similar_threshold)[0].cpu().numpy()

        # Save to JSONL file
        with open(args.output_path, "a") as f:
            for idx in idx_over_threshold:
                f.write(json.dumps(batch[idx]) + "\n")

        item_count += idx_over_threshold.shape[0]

    print(f"Saved {item_count} items to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence embeddings for astronomy examples.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the sentence transformer model.")
    parser.add_argument("--examples_path", type=str, required=True, help="Path to the examples JSON file.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the documents JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for model downloads.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to use, if any.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation.")
    parser.add_argument("--similar_threshold", type=float, default=0.8, help="Similarity threshold for filtering documents.")

    args = parser.parse_args()
    main(args)
