import numpy as np
import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import torch

nltk.download('brown')

def build_cooccurrence_matrix(corpus, window_size=5, min_count=5):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    filtered_words = [word for word, count in word_counts.items() if count >= min_count]
    print(len(filtered_words))
    word_to_id = {"<pad>": 0, "<unk>": 1}
    print(word_to_id.get("<unk>"))
    for word in filtered_words:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    V = len(word_to_id)
    cooccurrence = defaultdict(Counter)
    
    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in word_to_id:
                word_id = word_to_id[word]
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(sentence))
                for j in range(start, end):
                    if i != j and sentence[j] in word_to_id:
                        cooccurrence[word_id][word_to_id[sentence[j]]] += 1

    row, col, data = [], [], []
    for word_id, context in cooccurrence.items():
        for context_id, count in context.items():
            row.append(word_id)
            col.append(context_id)
            data.append(count)

    cooccurrence_matrix = csr_matrix((data, (row, col)), shape=(V, V), dtype=np.float32)
    return cooccurrence_matrix, word_to_id, id_to_word

def compute_svd_embeddings(cooccurrence_matrix, k=100):
    u, s, vt = svds(cooccurrence_matrix, k=k)
    idx = np.argsort(s)[::-1]
    u = u[:, idx]
    s = s[idx]
    embeddings = u * np.sqrt(s)
    return embeddings

if __name__ == "__main__":
    corpus = [[word.lower() for word in sentence] for sentence in brown.sents()]

    print("Building co-occurrence matrix...")
    cooccurrence_matrix, word_to_id, id_to_word = build_cooccurrence_matrix(corpus, window_size=5, min_count=5)

    print("Performing SVD...")
    embeddings = compute_svd_embeddings(cooccurrence_matrix, k=100)

    print("Saving embeddings...")
    torch.save({'embeddings': embeddings, 'word_to_id': word_to_id, 'id_to_word': id_to_word}, "svd.pt")

    print("SVD word embeddings saved to svd.pt")
