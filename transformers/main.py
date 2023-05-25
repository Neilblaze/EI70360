import math
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.3):

        super().__init__()

        self.d = d_model//n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.linear_Qs = nn.ModuleList([nn.Linear(d_model,self.d) for _ in range(n_heads)])
        self.linear_Ks = nn.ModuleList([nn.Linear(d_model,self.d) for _ in range(n_heads)])
        self.linear_Vs = nn.ModuleList([nn.Linear(d_model,self.d) for _ in range(n_heads)])

        self.dropout = nn.Dropout(dropout)
        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):

        QK = Q @ K.permute(0, 2, 1)

        scaled_QK = QK / math.sqrt(self.d)

        if mask is not None:
            scaled_QK = scaled_QK.masked_fill(mask==0, -1e9)

        attention_weights = F.softmax(scaled_QK, dim=-1)        

        output = attention_weights @ V      

        return output

    def forward(self, pre_q, pre_k, pre_v, mask=None):

        Qs = [linear_Q(pre_q) for linear_Q in self.linear_Qs]
        Ks = [linear_K(pre_k) for linear_K in self.linear_Ks]
        Vs = [linear_V(pre_v) for linear_V in self.linear_Vs]

        output_per_head = []

        for Q, K, V in zip(Qs, Ks, Vs):
            output = self.scaled_dot_product_attention(Q, K, V, mask)
            output_per_head.append(output)

        output = torch.cat(output_per_head, -1)

        projection = self.dropout(self.mha_linear(output))

        return projection

class ResidualLayerNorm(nn.Module):

    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.layer_norm(self.dropout(x)+residual)

class PWFFN(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()

        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.pwffn = PWFFN(d_model, d_ff, dropout)

    def forward(self, x):
        mha = self.mha(x,x,x)
        norm_1 = self.norm_1(mha, x)
        pwffn = self.pwffn(norm_1)
        norm_2 = self.norm_2(pwffn, norm_1)
        return norm_2

class Embeddings(nn.Module):

    def __init__(self, vocab_size, padding_idx, d_model):
        super().__init__()
        self.d_model=d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        embedding = self.embed(x)
        return embedding * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, d_model, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10000, (two_i/torch.Tensor([d_model]))).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:, :x.shape[1]].detach()
        x += pe
        return self.dropout(x)

class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        padding_idx,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.3
        ):
        super().__init__()

        self.embedding = Embeddings(vocab_size, padding_idx, d_model)
        self.PE = PositionalEncoding(max_seq_len, d_model, dropout)

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
        ) for layer in range(num_layers)])

    def forward(self, x):
        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)

        for encoder in self.encoders:
            encoding = encoder(encoding)

        return encoding

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super().__init__()

        self.norm_1 = ResidualLayerNorm(d_model, dropout)
        self.norm_2 = ResidualLayerNorm(d_model, dropout)
        self.norm_3 = ResidualLayerNorm(d_model, dropout)
        self.masked_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_dec_mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.pwffn = PWFFN(d_model, d_ff, dropout)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        masked_mha = self.masked_mha(x,x,x, mask=targ_mask)
        norm_1 = self.norm_1(masked_mha, x)
        enc_dec_mha = self.enc_dec_mha(norm_1, encoder_outputs, encoder_outputs, mask=src_mask)
        norm_2 = self.norm_2(enc_dec_mha, norm_1)
        pwffn = self.pwffn(norm_2)
        norm_3 = self.norm_3(pwffn, norm_2)
        return norm_3

class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        padding_idx,
        d_model,
        max_seq_len,
        num_heads,
        d_ff,
        num_layers,
        dropout=0.3
        ):
        super().__init__()

        self.embedding = Embeddings(vocab_size, padding_idx, d_model)
        self.PE = PositionalEncoding(max_seq_len, d_model, dropout)

        self.decoders = nn.ModuleList([DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
        ) for layer in range(num_layers)])

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)

        for decoder in self.decoders:
            encoding = decoder(encoding, encoder_outputs, trg_mask, src_mask)

        return encoding

if __name__=="__main__":

    toy_tokenized_inputs = torch.LongTensor([[1, 2, 3, 4, 0, 0]])
    print("Toy Tokenized Inputs: \n", toy_tokenized_inputs)
    print("Toy Tokenized Inputs Shape: \n", toy_tokenized_inputs.shape)

    toy_encoder = Encoder(vocab_size=5, padding_idx=0, d_model=4, max_seq_len=128, num_heads=2, d_ff=16, num_layers=2)
    toy_encoder_output = toy_encoder(toy_tokenized_inputs)
    print("Toy Encoder Output: \n", toy_encoder_output)
    print("Toy Encoder Output Shape: \n", toy_encoder_output.shape)