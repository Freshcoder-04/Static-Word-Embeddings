import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from nltk.corpus import brown
from tqdm import tqdm

def build_vocab(sentences, min_count=5):
    freq = {}
    for sentence in sentences:
        for token in sentence:
            token = token.lower()
            freq[token] = freq.get(token, 0) + 1

    vocab = {'<pad>': 0, '<unk>': 1}
    for token, count in freq.items():
        if count >= min_count and token not in vocab:
            vocab[token] = len(vocab)
    return vocab, freq

class CBOWDataset(Dataset):
    def __init__(self, sentences, vocab, window_size=2):
        self.samples = []
        unk_idx = vocab.get('<unk>')
        for sentence in sentences:
            indices = [vocab.get(token.lower(), unk_idx) for token in sentence]
            for i in range(len(indices)):
                target = indices[i]
                context = []
                for j in range(max(0, i - window_size), i):
                    context.append(indices[j])
                for j in range(i + 1, min(len(indices), i + window_size + 1)):
                    context.append(indices[j])
                if context:
                    self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    contexts = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([len(context) for context in contexts], dtype=torch.float)
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)
    return contexts_padded, lengths, targets

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx=0):
        super(CBOWModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        with torch.no_grad():
            self.in_embed.weight.data[pad_idx].fill_(0)
        self.out_embed.weight.data.zero_()

    def forward(self, contexts, lengths, targets, neg_samples):
        context_embeds = self.in_embed(contexts)  # (B, L, D)
        summed = context_embeds.sum(dim=1)          # (B, D)
        avg_context = summed / lengths.unsqueeze(1) # (B, D)

        pos_embed = self.out_embed(targets)         # (B, D)
        pos_score = torch.sum(avg_context * pos_embed, dim=1)  # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_embed = self.out_embed(neg_samples)     # (B, K, D)
        neg_score = torch.bmm(neg_embed, avg_context.unsqueeze(2)).squeeze(2)  # (B, K)
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
    
    dataset = CBOWDataset(sentences, vocab, window_size=2)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2, collate_fn=collate_fn)

    neg_weights = np.zeros(vocab_size)
    for word, idx in vocab.items():
        neg_weights[idx] = freq.get(word, 1) ** 0.75
    neg_weights = neg_weights / neg_weights.sum()
    neg_sampling_weights = torch.tensor(neg_weights, dtype=torch.float)

    embedding_dim = 100
    negative_samples = 5
    epochs = 10
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CBOWModel(vocab_size, embedding_dim, pad_idx=vocab['<pad>']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for contexts, lengths, targets in progress_bar:
            contexts = contexts.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            batch_size = targets.size(0)
            neg_samples = torch.multinomial(
                neg_sampling_weights,
                num_samples=batch_size * negative_samples,
                replacement=True
            ).view(batch_size, negative_samples).to(device)
            optimizer.zero_grad()
            loss = model(contexts, lengths, targets, neg_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    embeddings = model.in_embed.weight.data.cpu().numpy()
    torch.save({"embeddings": embeddings, "word_to_id": vocab, "id_to_word": id_to_word}, "./embeddings/cbow.pt")
    print("Trained embeddings and vocabulary saved to cbow.pt")

if __name__ == "__main__":
    main()
