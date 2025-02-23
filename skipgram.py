import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.corpus import brown
from tqdm import tqdm

def build_vocab(sentences, min_count=5):
    freq = {}
    for sentence in sentences:
        for token in sentence:
            token = token.lower()  # force lower-case
            freq[token] = freq.get(token, 0) + 1

    vocab = {}
    for token, count in freq.items():
        if count >= min_count:
            vocab[token] = len(vocab)
    if '<unk>' not in vocab:
        vocab['<unk>'] = len(vocab)
    return vocab, freq

class SkipGramDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=2):
        self.pairs = []
        self.vocab = vocab
        unk_idx = vocab.get('<unk>')
        for sentence in sentences:
            indices = [vocab.get(token.lower(), unk_idx) for token in sentence]
            for i, token in enumerate(indices):
                center_idx = token
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    self.pairs.append((center_idx, indices[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        self.init_embeddings()

    def init_embeddings(self):
        initrange = 0.5 / self.in_embed.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()

    def forward(self, center, pos_context, neg_context):
        center_embed = self.in_embed(center)  # (B, D)
        pos_embed = self.out_embed(pos_context) # (B, D)
        neg_embed = self.out_embed(neg_context) # (B, K, D)

        pos_score = torch.sum(center_embed * pos_embed, dim=1)  # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)

        loss = - (pos_loss + neg_loss)
        return loss.mean()

def main():
    sentences = [[word.lower() for word in sentence] for sentence in brown.sents()]
    print(f"Number of sentences: {len(sentences)}")

    vocab, freq = build_vocab(sentences, min_count=5)
    vocab_size = len(vocab)
    id_to_word = {idx: word for word, idx in vocab.items()}
    print(f"Vocab size: {vocab_size}")

    dataset = SkipGramDataset(sentences, vocab, window_size=2)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)

    neg_weights = np.zeros(vocab_size)
    for token, idx in vocab.items():
        neg_weights[idx] = freq.get(token, 1) ** 0.75
    neg_weights = neg_weights / neg_weights.sum()
    neg_sampling_weights = torch.tensor(neg_weights, dtype=torch.float)

    embedding_dim = 100
    negative_samples = 5
    epochs = 10
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for center, pos_context in progress_bar:
            center = center.to(device)
            pos_context = pos_context.to(device)
            batch_size = center.size(0)
            neg_context = torch.multinomial(neg_sampling_weights, num_samples=batch_size * negative_samples, replacement=True).view(batch_size, negative_samples).to(device)
            optimizer.zero_grad()
            loss = model(center, pos_context, neg_context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    embeddings = model.in_embed.weight.data.cpu().numpy()
    torch.save({"embeddings": embeddings, "word_to_id": vocab, "id_to_word": id_to_word}, "./embeddings/skipgram.pt")
    print("Trained embeddings and vocabulary saved to skipgram.pt")

if __name__ == '__main__':
    main()
