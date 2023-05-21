import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, d):

    QK = Q @ K.T

    scaled_QK = QK / math.sqrt(d)

    attention_weights = F.softmax(scaled_QK, dim=-1)

    output = attention_weights @ V

    return output, attention_weights

if __name__=="__main__":

    k = torch.Tensor([
        [10,0,0],
        [0,10,0],
        [0,0,10],
        [0,0,10],
    ])

    v = torch.Tensor([
        [1,0,1],
        [10,0,2],
        [100,5,0],
        [1000,6,0],
    ])

    q = torch.Tensor([[0,10,0],[0,0,10],[10,10,0]])

    out, attn = scaled_dot_product_attention(q, k, v, d=4)
    print('Attention weights:',attn.round(decimals=3))
    print('Output:',out.round(decimals=3))