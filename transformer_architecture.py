import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        # According to paper:
        # embed_size(d_model) = 512
        # heads(h) = 8
        # So, head_dim(d_model/h) = 512/8 = 64
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Checking if (d_model/h)*h = d_model
        assert self.head_dim * heads == embed_size

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        # N stores the no. of rows in Q vector
        N = queries.shape[0]

        # These variables stores the no. of columns of V, K, Q
        value_len, key_len, query_len = (
            values.shape[1],
            keys.shape[1],
            queries.shape[1],
        )

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # For multi-head attention, the V, K, Q vectors are split into N parts
        # That is 8 parts
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Using einsum for flexibility and manual broadcasting
        # It is performing Q.K^T (Q multiplied to the transpose of K)
        # Here n: batch size; q, k: query, key length; h: heads; d: head_dim
        # It says multiply q & k matching n & h, summing over d
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        # For making the model causal
        if mask is not None:
            mask = mask.bool()
            attention_scores = attention_scores.masked_fill(~mask, float("-1e20"))

        # Here it is basically doing Softmax((Q.K^T)/sqrt(d_k))
        # Which is head_dim wrt K, but here all Q, K, V has same head_sim
        attention = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        attention = self.attn_dropout(attention)

        # Here n: batch size; h: heads; q,v: query, value length; d: head_dim
        # It says multiply attention and values matching n & h, summing over v
        # This is basically doing Softmax((Q.K^T)/sqrt(d_k))*V
        out = torch.einsum("nhqv,nvhd->nqhd", attention, values)

        # Previously for multi-head attention, it was split into N parts
        # Now all those N parts are being concatenated
        out = out.reshape(N, query_len, self.embed_size)

        # This is a layer to average the information from all heads(W^0 from the paper)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        # Here, attention is the SelfAttention class
        # norm1 & norm2 are Layer Normalization methods for Add & Norm
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # In paper, for position-wise FFN, the inner-layer has a
        # dimensionality d_ff = 2048, which is 4*d_model(embed_size)
        # And a ReLU is used in between
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Implements the residual connection b/w attention & Q
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        # Implements the second residual connection b/w the
        # output of 1st Add & Norm sub-layer and output of FFN       
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        # Here the device is assigned
        # word_embedding stores the embedding of src_vocab_size
        # & each emdedding is of embed_size (d_model) size
        # Similarly positional_embedding stores embedding of max_length
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        # For implementing num_layers(Nx) no. of TransformerBlock as per the paper
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # Creates the positional indices from 0 to seq_len-1
        # & then copies the same vector N times, 
        # so that each batch gets its own positional indices
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # This creates the entry point into the encoder by adding 
        # input embedding with positional encoding created by the position indices
        # IMPORTANT-> Although the original paper used sin & cos to make positional embeddings
        # but they specifically stated that using learned positional embeddings produces
        # nearly identical result, so we are relying on nn.Embedding method
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        # Pass the data through N layers
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        # Assign SelfAttention, LayerNorm, TransformerBlock
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout)
        self.norm = nn.LayerNorm(embed_size)
        # Reusing the TransformerBlock class used in encoder
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        # Implements masked self-attention
        # tgt_mask makes the model causal
        attention = self.attention(x, x, x, tgt_mask)
        # It implements a residual connection between attention & x
        query = self.dropout(self.norm(attention + x))
        # Implements cross-attention
        # Here value & key comes from encoder, but query comes from decoder
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        # Here the device is assigned
        # word_embedding stores the embedding of tgt_vocab_size
        # & each emdedding is of embed_size (d_model) size
        # Similarly positional_embedding stores embedding of max_length
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

         # For implementing num_layers(Nx) no. of DecoderBlock as per the paper
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        # This is the top layer
        # It projects the final vector size to tgt_vocab_size
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        # Creates the positional indices from 0 to seq_len-1
        # & then copies the same vector N times, 
        # so that each batch gets its own positional indices
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # This creates the entry point into the encoder by adding 
        # input embedding with positional encoding created by the position indices
        x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))

        # Pass the data through N layers
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)

        # Returns the logits
        return self.fc_out(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, embed_size=512, num_layers=6, forward_expansion=4,
        heads=8, dropout=0.1, device="cpu", max_length=100):
        # Initializes the encoder & decoder along with src & tgt pad_idx & device
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Adds extra dimension at 1 & 2 dim
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # Adds a matrix of 1s in lower triangular part
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool().to(self.device)
        return tgt_pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)
