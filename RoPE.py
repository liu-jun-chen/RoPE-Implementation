import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, embd_size, max_len, device):
        super(RotaryPositionEmbedding, self).__init__()
        self.embd_size = embd_size
        self.max_len = max_len
        self.device = device
        # self.theta = torch.exp(-2 * (torch.arange(embd_size) // 2) / embd_size * math.log(10000.0))
        self.theta = 10000 ** (-2 * (torch.arange(0, embd_size, 2)) / embd_size)
    
    def forward(self, x):
        # x:[batch_size, seq_len, embd_size]
        batch_size, seq_len, _ = x.shape
        # position_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).to(self.device)

        # rotary_matrix = torch.zeros(batch_size, seq_len, self.embd_size, 2).to(self.device)

        idx_theta = torch.einsum('n,d->nd', torch.arange(seq_len), self.theta).to(self.device)
        sin, cos = idx_theta.sin(), idx_theta.cos()
        sin, cos = sin.unsqueeze(0).repeat(batch_size, 1, 1), cos.unsqueeze(0).repeat(batch_size, 1, 1)

        x_odd, x_even = x[..., ::2], x[..., 1::2]
        # x_partial_inverse = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        # queries_pos = x * cos + x_partial_inverse * sin
        x_pos_1 = x_odd * cos - x_even * sin
        x_pos_2 = x_odd * sin + x_even * cos

        x_pos = torch.zeros_like(x)
        x_pos[..., ::2] = x_pos_1
        x_pos[..., 1::2] = x_pos_2
        return x_pos
    
class SelfAttention(nn.Module):
    def __init__(self, embd_size, heads, dropout, device):
        super(SelfAttention, self).__init__()
        
        self.embd_size = embd_size
        self.heads = heads
        self.head_dim = embd_size // heads
        assert (self.head_dim * heads == embd_size), "Embedding size need to be divided by heads"

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False, device=device)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False, device=device)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False, device=device)
        self.fc_out = nn.Linear(self.embd_size, self.embd_size, device=device)
        self.dropout = nn.Dropout(dropout)

        self.rope_embedding = RotaryPositionEmbedding(self.head_dim, max_len=100, device=device)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, queries, keys, values, mask):
        batch_size, seq_len, _ = queries.shape
        q, k = self.rope_embedding(queries), self.rope_embedding(keys)

        values = values.reshape(batch_size, seq_len, self.heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.heads, self.head_dim)
        q = q.reshape(batch_size, seq_len, self.heads, self.head_dim)

        values = self.values(values)
        k = self.keys(k)
        q = self.queries(q)

        energy = torch.einsum('nqhd,nkhd->nhqk', q, k) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attn = torch.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('nhql, nlhd->nqhd', [attn, values]).reshape(
            batch_size, seq_len, self.heads*self.head_dim
        )
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 3, 4).to(device)
    # model = RotaryPositionEmbedding(4, 10, device)
    attention_model = SelfAttention(4, 1, 0, device)
    out = attention_model(x, x, x, None)
    print(out)    