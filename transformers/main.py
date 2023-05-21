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

if __name__=="__main__":

    toy_encodings = torch.Tensor([
        [
            [0.0, 0.1, 0.2, 0.3], 
            [1.0, 1.1, 1.2, 1.3], 
            [2.0, 2.1, 2.2, 2.3],
        ]
    ]) 
    print("Toy Encodings: \n", toy_encodings)
    print("Toy Encodings Shape: \n", toy_encodings.shape)

    toy_MHA_layer = MultiHeadAttention(d_model=4, n_heads=2)
    toy_MHA = toy_MHA_layer(toy_encodings)
    print("Toy MHA: \n", toy_MHA)
    print("Toy MHA Shape: \n", toy_MHA.shape)