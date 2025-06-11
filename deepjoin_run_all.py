import os
import pandas as pd
import pickle
import numpy as np
import time
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
import faiss
from Forward1 import process_onedataset

timing_report = {}

def column_to_text(filename, column_name, cells):
    cells = list(pd.Series(cells).dropna().unique())
    if not cells:
        return None
    max_len = max(len(c) for c in cells)
    min_len = min(len(c) for c in cells)
    avg_len = sum(len(c) for c in cells) / len(cells)
    preview = ', '.join(cells[:30])
    return f"{filename}. {column_name} contains {len(cells)} values ({max_len}, {min_len}, {avg_len:.1f}): {preview}"

def convert_csv_dir_to_pkl(csv_dir, out_pkl):
    start = time.time()
    data = []
    for fname in tqdm(os.listdir(csv_dir), desc="Converting seller CSVs to pkl"):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)

        # Attempt to read with UTF-8, then Latin-1; on parser errors, switch to python engine and skip bad lines
        try:
            df = pd.read_csv(path, dtype=str)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, dtype=str, encoding="latin1")
            except Exception as e:
                print(f"⚠️ Skipped {fname}: {e}")
                continue
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(
                    path,
                    dtype=str,
                    encoding="latin1",
                    engine="python",
                    on_bad_lines="skip"  # pandas >=1.3
                )
            except Exception as e:
                print(f"⚠️ Skipped {fname}: {e}")
                continue
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")
            continue

        # Convert each column to text prompt
        for col in df.columns:
            text = column_to_text(fname, col, df[col])
            if text:
                data.append(((fname, col), text))

    # Write out all collected prompts
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)

    ms = (time.time() - start) * 1000
    print(f"✅ Saved seller data to: {out_pkl}")
    print(f"⏱️ Seller CSV → pkl time: {ms:.1f} ms")
    timing_report["Seller CSV → pkl"] = ms


def convert_query_to_pkl(query_csv, column_name, out_pkl):
    start = time.time()
    df = pd.read_csv(query_csv, dtype=str)
    cells = df[column_name].dropna().unique()
    max_len = max(len(c) for c in cells)
    min_len = min(len(c) for c in cells)
    avg_len = sum(len(c) for c in cells) / len(cells)
    preview = ', '.join(cells[:30])
    desc = f"{os.path.basename(query_csv)}. {column_name} contains {len(cells)} values ({max_len}, {min_len}, {avg_len:.1f}): {preview}"
    with open(out_pkl, "wb") as f:
        pickle.dump([((os.path.basename(query_csv), column_name), desc)], f)
    ms = (time.time() - start) * 1000
    print(f"✅ Saved query column to: {out_pkl}")
    print(f"⏱️ Query CSV → pkl time: {ms:.1f} ms")
    timing_report["Query CSV → pkl"] = ms

# def embed_text_pkl(in_pkl, out_pkl, model_path):
#     start = time.time()
#     with open(in_pkl, "rb") as f:
#         data = pickle.load(f)
#     model = SentenceTransformer(model_path)
#     embedded = []
#     for key, text in tqdm(data, desc=f"Embedding {os.path.basename(in_pkl)}"):
#         vec = model.encode(text)
#         embedded.append((key, np.array(vec)))
#     with open(out_pkl, "wb") as f:
#         pickle.dump(embedded, f)
#     ms = (time.time() - start) * 1000
#     print(f"✅ Saved embeddings to: {out_pkl}")
#     print(f"⏱️ Embedding time for {os.path.basename(in_pkl)}: {ms:.1f} ms")
#     timing_report[f"Embedding {os.path.basename(in_pkl)}"] = ms

def match_topk(query_pkl, seller_pkl, query_col, topk):
    start = time.time()
    with open(query_pkl, "rb") as f:
        query_data = pickle.load(f)
    with open(seller_pkl, "rb") as f:
        seller_data = pickle.load(f)
    query_filtered = [(k, v) for (k, v) in query_data if k[1].lower() == query_col.lower()]
    if not query_filtered:
        raise ValueError(f"❌ Column '{query_col}' not found in {query_pkl}")
    q_key, q_vec = query_filtered[0]
    s_keys = [k for k, _ in seller_data]
    s_vecs = np.stack([v for _, v in seller_data])

    index = faiss.IndexHNSWFlat(s_vecs.shape[1], 32)
    index.hnsw.efSearch = 128
    index.hnsw.efConstruction = 200
    index.add(s_vecs)

    D, I = index.search(np.array([q_vec]), topk)

    print(f"\n🔍 Query column: {q_key}")
    print(f"\n🎯 Top-{topk} Joinable Columns:")
    for i, idx in enumerate(I[0]):
        print(f"{i+1}. {s_keys[idx]} (distance: {D[0][i]:.4f})")

    ms = (time.time() - start) * 1000
    print(f"⏱️ Matching time: {ms:.1f} ms")
    timing_report["Matching"] = ms

def print_report():
    print("\n📝 Timing Summary Report:")
    for key, val in timing_report.items():
        print(f"• {key.ljust(30)}: {val:.1f} ms")
    total = sum(timing_report.values())
    print(f"\n⏱️ Total pipeline time       : {total:.1f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full DeepJoin pipeline (query + datalake)")
    parser.add_argument("--query_csv", required=True, help="Query CSV file path")
    parser.add_argument("--query_col", required=True, help="Column name to use from query CSV")
    parser.add_argument("--datalake_dir", required=True, help="Directory containing many seller CSVs")
    parser.add_argument("--topk", type=int, default=10, help="Top-K joinable columns to return")
    parser.add_argument("--model_path", default="output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27", help="Path to fine-tuned DeepJoin model")

    args = parser.parse_args()  # ← ✅ THIS LINE FIXES THE ERROR

    total_start = time.time()

    os.makedirs("infer_input", exist_ok=True)
    os.makedirs("infer_output", exist_ok=True)

    seller_txt_pkl = "infer_input/seller.pkl"
    query_txt_pkl = "infer_input/query.pkl"
    seller_vec_pkl = "infer_output/seller.pkl"
    query_vec_pkl = "infer_output/query.pkl"

    convert_csv_dir_to_pkl(args.datalake_dir, seller_txt_pkl)
    convert_query_to_pkl(args.query_csv, args.query_col, query_txt_pkl)

    seller_ms = process_onedataset(seller_txt_pkl, model_name=args.model_path, storepath="infer_output/")
    timing_report["Embedding seller.pkl"] = seller_ms


    query_ms = process_onedataset(query_txt_pkl, model_name=args.model_path, storepath="infer_output/")
    timing_report["Embedding query.pkl"] = query_ms

    match_topk(query_vec_pkl, seller_vec_pkl, args.query_col, args.topk)

    print_report()

