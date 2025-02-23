import sys
import torch
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import brown
from scipy.stats import spearmanr
from wordsim import load_embeddings, compute_cosine_similarity

def main():
    embeddings_path = ['embeddings/svd.pt', 'embeddings/cbow.pt', 'embeddings/skipgram.pt']

    vocab_svd, embeddings_svd = load_embeddings(embeddings_path[0])
    vocab_cbow, embeddings_cbow = load_embeddings(embeddings_path[1])
    vocab_skipgram, embeddings_skipgram = load_embeddings(embeddings_path[2])
    
    vocabs, all_embeddings = [vocab_svd, vocab_cbow, vocab_skipgram], [embeddings_svd, embeddings_cbow, embeddings_skipgram]

    try:
        wordsim_df = pd.read_csv("wordsim353.csv")
    except Exception as e:
        print("Error loading wordsim353.csv:", e)
        sys.exit(1)

    human_col = 'Human (Mean)'

    names = ['svd', 'cbow', 'skipgram']
    spearman = []
    for i in range(0,3):
        vocab = vocabs[i]
        embeddings = all_embeddings[i]

        lower_vocab = {token.lower(): idx for token, idx in vocab.items()}

        human_scores = []
        cosine_scores = []

        for _, row in wordsim_df.iterrows():
            w1 = str(row["Word 1"]).lower()
            w2 = str(row["Word 2"]).lower()
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
        spearman.append(spearman_corr)

        plt.figure(figsize=(6,6))
        plt.scatter(human_scores, cosine_scores, alpha=0.5, color='blue')
        plt.xlabel("Human (Mean) Similarity")
        plt.ylabel("Cosine Similarity")
        plt.title(f"{names[i]} Cosine Similarity vs Human (Mean)")
        plt.plot([min(human_scores), max(human_scores)],
                [min(human_scores), max(human_scores)], 'r--', label="y=x")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./images/{names[i]}_cosine_vs_human.png")
        # plt.show()
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.bar(names, spearman, color=['blue', 'green', 'orange'])
    plt.ylabel("Spearman's Rank Correlation")
    plt.title("Comparison of Spearman's Rank Correlation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("./images/spearman_comparison.png")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()