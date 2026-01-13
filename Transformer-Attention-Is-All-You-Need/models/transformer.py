import torch
import torch.nn as nn
import torch.nn.funtional as F
from layers import EncoderBlock, DecoderBlock
from embedding import PositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, enc_block=6, dec_block=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.positionalembedding = PositionalEmbedding()

        self.enc_stack = nn.ModuleList([EncoderBlock() for _ in range(enc_block)])
        self.dec_stack = nn.ModuleList([DecoderBlock() for _ in range(dec_block)])

        self.linear_projection = nn.Linear(512, vocab_size)

    def forward(self, enc_input, dec_input):
        enc_input = self.positionalembedding(self.embedding(enc_input) * (512 ** 0.5))
        dec_input = self.positionalembedding(self.embedding(dec_input) * (512 ** 0.5))
        for block in self.enc_stack:
            enc_input = block(enc_input)
        for block in self.dec_stack:
            dec_input = block(dec_input, enc_input)

        final_output = self.linear_projection(dec_input)

        return final_output
