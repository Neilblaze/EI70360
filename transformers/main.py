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

    def scaled_dot_product_attention(self, Q, K, V):

        QK = Q @ K.permute(0, 2, 1)

        scaled_QK = QK / math.sqrt(self.d)

        attention_weights = F.softmax(scaled_QK, dim=-1)        

        output = attention_weights @ V      

        return output

    def forward(self, x):

        Qs = [linear_Q(x) for linear_Q in self.linear_Qs]
        Ks = [linear_K(x) for linear_K in self.linear_Ks]
        Vs = [linear_V(x) for linear_V in self.linear_Vs]

        output_per_head = []

        for Q, K, V in zip(Qs, Ks, Vs):
            output = self.scaled_dot_product_attention(Q, K, V)
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
        mha = self.mha(x)
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

if __name__=="__main__":

    toy_tokenized_inputs = torch.LongTensor([[1, 2, 3, 4, 0, 0]])
    print("Toy Tokenized Inputs: \n", toy_tokenized_inputs)
    print("Toy Tokenized Inputs Shape: \n", toy_tokenized_inputs.shape)

    toy_embeddings_layer = Embeddings(5, 0, 4)
    toy_embeddings = toy_embeddings_layer(toy_tokenized_inputs)
    print("Toy Embeddings: \n", toy_embeddings)
    print("Toy Embeddings Shape: \n", toy_embeddings.shape)

    toy_PE_layer = PositionalEncoding(128, 4)
    toy_PE = toy_PE_layer(toy_embeddings)
    print("Toy PE: \n", toy_PE)
    print("Toy PE Shape: \n", toy_PE.shape)

    toy_encoder_layer = EncoderLayer(d_model=4, n_heads=2, d_ff=16)
    toy_encoder_layer_output = toy_encoder_layer(toy_PE)
    print("Toy Encoder Layer Output: \n", toy_encoder_layer_output)
    print("Toy Encoder Layer Output Shape: \n", toy_encoder_layer_output.shape)