import torch.nn as nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)