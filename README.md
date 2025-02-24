# Static-Word-Embeddings

Below is an example of a README.md template for your repository:

# Static Word Embeddings

This repository contains implementations of three methods for learning static word embeddings on the Brown corpus:

- **SVD-based Embeddings**: Constructs a word co-occurrence matrix and performs truncated SVD.
- **CBOW (Continuous Bag-of-Words)**: Trains a neural embedding model using negative sampling.
- **Skip-Gram**: Trains a predictive model using negative sampling.

Embeddings are saved as `.pt` files and can be evaluated using the WordSim-353 dataset.

---

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Pandas
- NLTK
- Matplotlib
- tqdm

Install the required packages using pip:

```bash
pip install torch numpy scipy pandas nltk matplotlib tqdm
```

---

## Setup

Download necessary NLTK data (the Brown corpus and Punkt tokenizer):

```bash
python -m nltk.downloader brown punkt
```

Place the WordSim-353 CSV file in the repository directory as `wordsim353.csv`.

---

## Files

- **svd.py**  
  Builds a word co-occurrence matrix from the Brown corpus and computes word embeddings using truncated SVD.  
  **Output:** `svd.pt`

- **cbow.py**  
  Trains a CBOW model with negative sampling on the Brown corpus.  
  **Output:** `cbow.pt`

- **skipgram.py**  
  Trains a Skip-Gram model with negative sampling on the Brown corpus.  
  **Output:** `skipgram.pt`

- **wordsim.py**  
  Evaluates a given embedding file on the WordSim-353 dataset by computing cosine similarities for word pairs and calculating Spearman's Rank Correlation.

- **analysis.py**  
  Loads embeddings from SVD, CBOW, and Skip-Gram models, computes their Spearman rank correlations on WordSim-353, and plots a bar chart comparing the three methods and also plots the cosine similarity vs Human (Mean) on a scatter plot for the 3 embeddings.

---

## Usage

### 1. Train Embeddings

Run the following scripts to generate the embeddings:

- **SVD:**
  ```bash
  python svd.py
  ```
  Generates `./embeddings/svd.pt`.

- **CBOW:**
  ```bash
  python cbow.py
  ```
  Generates `./embeddings/cbow.pt`.

- **Skip-Gram:**
  ```bash
  python skipgram.py
  ```
  Generates `./embeddings/skipgram.pt`.

### 2. Evaluate Word Similarity

To evaluate a specific model's embeddings on the WordSim-353 dataset, run:
  
```bash
python wordsim.py <embedding_file>.pt
```

For example, to evaluate the CBOW embeddings:
  
```bash
python wordsim.py ./embeddings/cbow.pt
```

Ensure that `wordsim353.csv` is in the repository directory.

### 3. Compare Models

To compare the Spearman Rank Correlations of the three methods and generate a comparison plot, run:

```bash
python analysis.py
```

This script will load `svd.pt`, `cbow.pt`, and `skipgram.pt` from the `./embeddings` directory, compute the correlations, and produce a bar chart (saved as `./images/spearman_comparison.png`).

---

## Data

- **Brown Corpus:** Downloaded automatically via NLTK.
- **WordSim-353:** Download the dataset from [kaggle](https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd) and place the CSV file in the repository directory as `wordsim353.csv`.
---

<p align="center" style="font-size: 40px;"><b>Thank You</b></p>

---