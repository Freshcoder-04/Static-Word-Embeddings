import sys
import torch
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import brown
from scipy.stats import spearmanr

nltk.download('brown')
nltk.download('punkt')


def load_embeddings(embedding_path):
    data = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "embeddings" in data and "word_to_id" in data:
        vocab = data["word_to_id"]
        embeddings = data["embeddings"]
        return vocab, embeddings

def compute_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def main():
    if len(sys.argv) != 2:
        print("Usage: python wordsim.py <embedding_path>.pt")
        sys.exit(1)
    embedding_path = sys.argv[1]
    vocab, embeddings = load_embeddings(embedding_path)

    try:
        wordsim_df = pd.read_csv("wordsim353.csv")
    except Exception as e:
        print("Error loading wordsim353.csv:", e)
        sys.exit(1)

    human_col = 'Human (Mean)'

    lower_vocab = {token.lower(): idx for token, idx in vocab.items()}

    human_scores = []
    cosine_scores = []

    for _, row in wordsim_df.iterrows():
        w1 = str(row["Word 1"]).lower()
        w2 = str(row["Word 2"]).lower()
        # print(w1,w2)
        if w1 in lower_vocab and w2 in lower_vocab:
            idx1 = lower_vocab[w1]
            idx2 = lower_vocab[w2]
            vec1 = embeddings[idx1]
            vec2 = embeddings[idx2]
            cos_sim = compute_cosine_similarity(vec1, vec2)
            human_scores.append(row[human_col])
            cosine_scores.append(cos_sim)
    print(len(human_scores),wordsim_df.shape)
    if len(human_scores) == 0:
        print("No valid word pairs found (words missing from vocabulary).")
        sys.exit(1)

    spearman_corr, _ = spearmanr(human_scores, cosine_scores)
    print(f"Spearman Rank Correlation: {spearman_corr:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(human_scores, cosine_scores, alpha=0.5, color='blue')
    plt.xlabel("Human (Mean) Similarity")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity vs Human (Mean)")
    plt.plot([min(human_scores), max(human_scores)],
             [min(human_scores), max(human_scores)], 'r--', label="y=x")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("cosine_vs_human.png")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
