import sys
import pickle
import numpy as np
import re
from pathlib import Path
from model import DeepTernaryNetworkMHot

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def main(input_file_str, dataset_dir_str, weights_dir_str, context_size=3):
    input_file = Path(input_file_str).expanduser()
    dataset_dir = Path(dataset_dir_str).expanduser()
    weights_dir = Path(weights_dir_str).expanduser()

    with open(input_file, 'r', encoding='utf-8') as f:
        words = clean_text(f.read())

    # Build Vocabulary
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # Create Dataset (Context -> Target)
    X, Y = [], []
    for i in range(len(words) - context_size):
        context = [word_to_id[w] for w in words[i:i+context_size]]
        target = word_to_id[words[i+context_size]]
        X.append(context)
        Y.append(target)

    # 80/20 Train-Test Split
    split_idx = int(len(X) * 0.8)
    dataset = {
        "X_train": X[:split_idx], "Y_train": Y[:split_idx],
        "X_test": X[split_idx:], "Y_test": Y[split_idx:],
        "word_to_id": word_to_id, "id_to_word": id_to_word,
        "context_size": context_size
    }

    dataset_file = dataset_dir / "dataset.pkl"
    with open(dataset_file, "wb") as f:
        pickle.dump(dataset, f)

    # Initialize Model [Input Vocab -> 64 -> 64 -> Output Vocab]
    model = DeepTernaryNetworkMHot(vocab_size, [64, 64], vocab_size)
    weights_file = weights_dir / "weights.pkl"
    model.save_weights(weights_file)
    
    print(f"Setup complete. Vocab size: {vocab_size}. Training samples: {len(X[:split_idx])}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run  src/ternary_bravo/setup_script.py <text_file> <dataset_dir> <weights_dir>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
