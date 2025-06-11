import pickle
import argparse
import numpy as np

def inspect_pkl_file(path, show=3):
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded {len(data)} entries from: {path}\n")

    for i, (key, vector) in enumerate(data[:show]):
        print(f"{i+1}. Column: {key}")
        if isinstance(vector, np.ndarray):
            print(f"   Vector shape: {vector.shape}")
            print(f"   First 5 dims: {vector[:5]}\n")
        elif isinstance(vector, str):
            print(f"   Description: {vector[:200]}...\n")
        else:
            print(f"   Value: {vector}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a DeepJoin-compatible .pkl file.")
    parser.add_argument("pkl_file", type=str, help="Path to .pkl file")
    parser.add_argument("--show", type=int, default=3, help="Number of entries to display")

    args = parser.parse_args()
    inspect_pkl_file(args.pkl_file, args.show)
