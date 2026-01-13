import torch
import torch.nn as nn
from attention import MultiheadAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention_block = MultiheadAttention()
        self.feedforward = FeedForwardNetwork()
        self.layernorm = nn.LayerNorm(512)

    def forward(self, x):
        attention_val = self.multihead_attention_block(x)
        x = x + attention_val
        x = self.layernorm(x)
        feed_val = self.feedforward(x)
        x = x + feed_val
        x = self.layernorm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_attention_block = MultiheadAttention(transition_state=True)
        self.masked_multihead_attention_block = MultiheadAttention(masked=True)
        self.feedforward = FeedForwardNetwork()
        self.layernorm = nn.LayerNorm(512)

    def forward(self, x, enc_output):
        attention_val = self.masked_multihead_attention_block(x)
        x = x + attention_val
        x = self.layernorm(x)
        attention_val = self.multihead_attention_block(x, enc_output)
        x = x + attention_val
        x = self.layernorm(x)
        feed_val = self.feedforward(x)
        x = x + feed_val
        x = self.layernorm(x)
        return x
