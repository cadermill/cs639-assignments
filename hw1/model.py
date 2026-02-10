import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size)).astype(np.float32)
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                emb[vocab[word]] = vector
    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.args.emb_size if i == 0 else self.args.hid_size, self.args.hid_size)
            for i in range(self.args.hid_layer)
        ])
        self.output_layer = nn.Linear(self.args.hid_size, self.tag_size)
        self.word_dropout = nn.Dropout(self.args.word_drop)
        self.embedding_dropout = nn.Dropout(self.args.emb_drop)
        self.hidden_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.08
        nn.init.uniform_(self.embedding.weight, -v, v)
        for layer in self.hidden_layers:
            nn.init.uniform_(layer.weight, -v, v)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.output_layer.weight, -v, v)
        nn.init.zeros_(self.output_layer.bias)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # Apply word dropout
        mask = (torch.rand(x.size(), device=x.device) > self.args.word_drop).float()
        x = x * mask.long()

        # Embed the input words
        embedded = self.embedding(x)  # [batch_size, seq_length, emb_size]
        embedded = self.embedding_dropout(embedded)

        # Compute the pooling method
        if self.args.pooling_method == "avg":
            pooled = embedded.mean(dim=1)  # [batch_size, emb_size]
        elif self.args.pooling_method == "sum":
            pooled = embedded.sum(dim=1)  # [batch_size, emb_size]
        elif self.args.pooling_method == "max":
            pooled, _ = embedded.max(dim=1)  # [batch_size, emb_size]

        # Pass through the hidden layers
        hidden = pooled
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
            hidden = self.hidden_dropout(hidden)

        # Compute the output scores
        scores = self.output_layer(hidden)  # [batch_size, tag_size]

        return scores
