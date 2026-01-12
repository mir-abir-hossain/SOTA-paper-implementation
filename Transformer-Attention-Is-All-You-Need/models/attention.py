import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Self_Attention(nn.Module):

    def __init__(self, dmodel: int=512, dk:int =512, masked: bool=False, transition_state: bool=False):
        super().__init__()
        self.dmodel, self.dk = dmodel, dk
        self.masked = masked
        self.query_weight = nn.Linear(self.dmodel, self.dk) # [B, 512, 512]
        self.key_weight = nn.Linear(self.dmodel, self.dk) # [B, 512, 512]
        self.value_weight = nn.Linear(self.dmodel, self.dk) # [B, 512, 512]
        self.transition_state = transition_state

    def forward(self, x, prev_output=None):
        query = self.query_weight(x) # [B,  word_count, 512]

        if self.transition_state:
            # Cross-attention: Keys and Values come from encoder (prev_output)
            key= self.key_weight(prev_output) # [B, word_count, 512]
            value = self.value_weight(prev_output) # [B, word_count, 512]
        else:
            # Self-attention: Keys and Values come from input x
            key= self.key_weight(x) # [B, word_count, 512]
            value = self.value_weight(x) # [B, word_count, 512]

        attention_score = torch.matmul(query, key.transpose(-1, -2)) # [B , word_count, word_count]
        attention_score = attention_score / (self.dk ** 0.5)
        if self.masked==True:
            sentence_length = x.size(-2)
            mask = torch.triu(torch.ones((sentence_length, sentence_length), device=x.device), diagonal=1).bool()
            attention_score = attention_score.masked_fill(mask, -1e9)

        attention_score = F.softmax(attention_score, -1)
        attention_score = torch.matmul(attention_score, value) # [B, word_count, 512]
        return attention_score


class MultiheadAttention(nn.Module):

    def __init__(self, head: int=8, embed_dim: int=512, masked: bool=False, transition_state: bool=False):
        super().__init__()
        self.masked = masked
        self.head = head
        self.embed_dim = embed_dim
        self.each_embed_dim = int(self.embed_dim/self.head)
        self.attention_heads = nn.ModuleList([Self_Attention(dk=self.each_embed_dim, masked=self.masked, 
                                                             transition_state=transition_state) for _ in range(head)])
        self.weight = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, prev_output=None):
        processed_chunks = {}
        for h in range(self.head):
            chunk_attention_block = self.attention_heads[h]
            processed_chunks[f"chunk_{h}"] = chunk_attention_block(x, prev_output)

        x = torch.concat(tuple(processed_chunks.values()), dim=-1)
        x = self.weight(x)
        print(x.shape)
        return x
