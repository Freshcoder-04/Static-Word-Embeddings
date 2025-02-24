import sys
import torch
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from wordsim import load_embeddings, compute_cosine_similarity

nltk.download('brown')
nltk.download('punkt')

def main():
    embeddings_paths = {
        'svd': 'embeddings/svd.pt',
        'cbow': 'embeddings/cbow.pt',
        'skipgram': 'embeddings/skipgram.pt'
    }

    vocabs = {}
    all_embeddings = {}
    lower_vocabs = {}
    for model_name, path in embeddings_paths.items():
        vocab, embeddings = load_embeddings(path)
        vocabs[model_name] = vocab
        all_embeddings[model_name] = embeddings
        lower_vocabs[model_name] = {token.lower(): idx for token, idx in vocab.items()}

    try:
        wordsim_df = pd.read_csv("wordsim353.csv")
    except Exception as e:
        print("Error loading wordsim353.csv:", e)
        sys.exit(1)

    human_col = 'Human (Mean)'
    wordsim_df['scaled_human'] = wordsim_df[human_col].apply(lambda x: round(2 * (x / 10.0) - 1, 3))

    wordsim_copy = wordsim_df.copy()

    for model in embeddings_paths.keys():
        wordsim_copy[f"{model}_cosine"] = np.nan

    for idx, row in wordsim_copy.iterrows():
        w1 = str(row["Word 1"]).lower()
        w2 = str(row["Word 2"]).lower()
        for model in embeddings_paths.keys():
            lower_vocab = lower_vocabs[model]
            embeddings = all_embeddings[model]
            if w1 in lower_vocab and w2 in lower_vocab:
                idx1 = lower_vocab[w1]
                idx2 = lower_vocab[w2]
                cos_sim = compute_cosine_similarity(embeddings[idx1], embeddings[idx2])
                wordsim_copy.at[idx, f"{model}_cosine"] = float(f"{cos_sim:.3f}")

    wordsim_copy.to_csv("Comparision.csv", index=False)
    print("Saved updated wordsim DataFrame to wordsim_with_all_cosines.csv")

    spearman_results = {}
    for model in embeddings_paths.keys():
        valid_rows = wordsim_copy.dropna(subset=[f"{model}_cosine"])
        human_scores = valid_rows['scaled_human'].tolist()
        cosine_scores = valid_rows[f"{model}_cosine"].tolist()
        if len(human_scores) == 0:
            print(f"No valid word pairs found for {model}.")
            sys.exit(1)
        spearman_corr, _ = spearmanr(human_scores, cosine_scores)
        spearman_results[model] = spearman_corr
        print(f"Spearman Rank Correlation for {model}: {spearman_corr:.4f}")

        plt.figure(figsize=(6,6))
        plt.scatter(human_scores, cosine_scores, alpha=0.5, color='blue')
        plt.xlabel("Scaled Human (Mean) Similarity [-1,1]")
        plt.ylabel("Cosine Similarity [-1,1]")
        plt.title(f"{model.upper()}: Cosine Similarity vs Scaled Human (Mean)")
        plt.plot([min(human_scores), max(human_scores)],
                 [min(human_scores), max(human_scores)], 'r--', label="y=x")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./images/{model}_cosine_vs_human.png")
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(spearman_results.keys(), spearman_results.values(), color=['blue', 'green', 'orange'])
    plt.ylabel("Spearman's Rank Correlation")
    plt.title("Comparison of Spearman's Rank Correlation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("./images/spearman_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
