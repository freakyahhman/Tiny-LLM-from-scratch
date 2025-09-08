import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

class ModelArgs:
    """
    Definition of Model's Arguments and Hypermeters.

    Attributes:
        max_batch_size (int): Maximum Batch size.
        max_seq_len (int): Maximum Sequence Length.
        dtype (torch.dtype): Data type for computation.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        hid_dim (int): Hidden layer dimension.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of query heads.
        n_kv_groups (int): Number of key and value groups.
        rope_theta (int): Base for rotary positional encoding.
    """
    max_batch_size: int = 4
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 50257
    embed_dim: int = 512
    hid_dim: int  = 2048
    n_layers: int = 12
    n_heads: int = 4
    n_kv_groups: int = 1
    rope_theta: int = 10000

class FeedForward(nn.Module):
    """
    FeedForward layer in the transformer blocks.

    Args: 
        embed_dim: Model dimension.
        hid_dim: hidden dimension.
        dtype: data type for computation.
    """
    def __init__(self, embed_dim=512, hid_dim=2048, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hid_dim, dtype=dtype)
        self.fc2 = nn.Linear(embed_dim, hid_dim, dtype=dtype)
        self.fc3 = nn.Linear(hid_dim, embed_dim, dtype=dtype)
    def forward(self, x):
        """
        Forward pass with gating mechanism.
        Input: x (torch.Tensor): the lastest token.
        Output: the final representation of the lastest token before being fed to the next layer.
        """
        return self.fc3(F.gelu(self.fc1(x)) * self.fc2(x))
    
class RMSNorm(nn.Module):
    """
    The Root Mean Square Normalization.

    Attributes:
        epsilon: a small real number to avoid dividing 0.
        dtype: data type for computation.
    """
    def __init__(self, epsilon=1e-6, dtype=torch.bfloat16, embed_dim=512):
        super().__init__()
        self.rms = nn.RMSNorm(eps=epsilon, dtype=dtype, normalized_shape=embed_dim)

    def forward(self, x):
        """
        Applying RMSNorm to x.
        Input: x (torch.Tensor).
        Output: normalized Tensor.
        """
        return self.rms(x)

def Computing_RoPE_params(theta: int, dim: int, seq_len: int, dtype=torch.bfloat16):
    """
    Computing the rotate angle for each embedding vector in the sequence.
    Args:
        theta: a parameter for adjusting the rotate speed.
        dim: dimension of the vectors.
        seq_len: the sequence length.
    Output: the computed params for each vector in the sequence.
    """
    freq = theta ** (-torch.arange(0, dim // 2) / (dim // 2))
    freq = freq.to(dtype)
    positions = freq[None, :] * torch.arange(seq_len)[:, None].to(dtype)
    return positions.cos(), positions.sin()

def Apply_RoPE(x: torch.Tensor, cos, sin):
    """
    Apply RoPE to queries or keys.
    Args:
        x:   [batch, head, seq_len, dim]
        cos: [seq_len, dim/2]
        sin: [seq_len, dim/2]
    Returns:
        x_rot: [batch, head, seq_len, dim]
    """
    x1, x2 = x[..., ::2], x[..., 1::2]  # [B, H, L, d_h/2]

    # match shape for broadcasting
    cos = cos[None, None, :, :]  # [1,1,L,d_h/2]
    sin = sin[None, None, :, :]

    x_rot = torch.stack([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """
    GroupedAttention module.
    Args:
        embed_dim: the embedding dimension.
        n_heads: the number of query heads.
        n_kv_groups: the number of key and value groups.
        dtype: the data type for computation.    
    """
    def __init__(self, embed_dim, n_heads=4, n_kv_groups=1, dtype=torch.bfloat16):
        super().__init__()
        self.dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.dtype=dtype
        self.W_q = nn.Linear(embed_dim, embed_dim * n_heads, dtype=dtype)
        self.W_k = nn.Linear(embed_dim, embed_dim * n_kv_groups, dtype=dtype)
        self.W_v = nn.Linear(embed_dim, n_kv_groups * embed_dim, dtype=dtype)
        self.W_out = nn.Linear(embed_dim * n_heads, embed_dim, dtype=dtype)
        
    def forward(self, x, theta):
        """
        Forward pass for multi-head self-attention with RoPE (Rotary Positional Embedding).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            theta (float): Base frequency for RoPE position encoding.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, n_heads * dim).

        Steps:
            1. Linear projection:
                - Compute queries (Q), keys (K), and values (V) via learned projections W_q, W_k, W_v.
            2. Reshape & permute:
                - Q -> (B, n_heads, L, D)
                - K, V -> (B, n_kv_groups, L, D)
            3. Apply causal mask:
                - Lower-triangular mask ensures each token attends only to previous tokens.
            4. Apply RoPE:
                - Rotary position embeddings are applied to Q and K.
            5. Expand K, V:
                - Repeat-interleave K and V so that their groups match n_heads.
            6. Attention scores:
                - Compute Q @ K^T / sqrt(D) with mask applied.
            7. Softmax:
                - Convert scores into attention probabilities.
            8. Weighted sum:
                - Multiply probabilities with V -> context vectors.
            9. Merge heads:
                - Rearrange (B, H, L, D) -> (B, L, H*D).
            10. Final projection:
                - Pass through W_out to get output of shape (B, L, H*D).
        """
        batch, seq_len, dim = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.view(batch, seq_len, self.n_heads, self.dim)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.view(batch, seq_len, self.n_kv_groups, self.dim)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.view(batch, seq_len, self.n_kv_groups, self.dim)
        v = v.permute(0, 2, 1, 3).contiguous()

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

        cos, sin = Computing_RoPE_params(theta=theta, dim=self.dim, seq_len=seq_len, dtype=self.dtype)
        cos, sin = cos.to(x.device), sin.to(x.device)
        q = Apply_RoPE(q, cos, sin)
        k = Apply_RoPE(k, cos, sin)

        k = k.repeat_interleave(self.n_heads // self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_groups, dim=1)


        attn_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)
        attn_score.masked_fill(mask == 0, float("-inf"))
        attn_prob = torch.softmax(attn_score, dim=-1)
        out = attn_prob @ v
        out = out.permute(0, 2, 1, 3).contiguous()

        out = out.view(batch, seq_len, self.n_heads * self.dim)

        out = self.W_out(out)
        return out

class Transformer(nn.Module):
    """
    A single Transformer block combining:
        - Grouped Query Attention (GQA)
        - FeedForward network (FFN)
        - RMSNorm normalization
        - Residual connections

    Args:
        args: Configuration object containing:
            - embed_dim (int): Embedding dimension.
            - n_heads (int): Number of attention heads.
            - n_kv_groups (int): Number of key-value groups (for GQA).
            - hid_dim (int): Hidden dimension of the FeedForward layer.
            - dtype (torch.dtype): Data type used in computations.

    Components:
        - self.attn (GroupedQueryAttention): Multi-head grouped attention module.
        - self.ff   (FeedForward): Position-wise feedforward layer.
        - self.norm (RMSNorm): Root-mean-square normalization.
    """
    def __init__(self, args):
        super().__init__()
        self.theta = args.rope_theta
        self.attn = GroupedQueryAttention(
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_kv_groups=args.n_kv_groups,
            dtype=args.dtype
        )
        self.ff = FeedForward(embed_dim=args.embed_dim, hid_dim=args.hid_dim, dtype=args.dtype)
        self.norm = RMSNorm()
    
    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, embed_dim).

        Steps:
            1. Normalize input x with RMSNorm.
            2. Apply grouped query attention with residual connection:
               x = Norm(x) → Attention(x) + x
            3. Normalize the result again.
            4. Apply feedforward network with residual connection:
               x = Norm(x) → FFN(x) + x
            5. Return final output.
        """
        x = self.norm(x)
        temp = x
        x = self.attn(x, theta=self.theta)
        x = self.norm(x + temp)
        temp = x
        x = self.ff(x)
        x = self.norm(x + temp)
        return x
    
class Model(nn.Module):
    """
    Transformer-based Language Model (GPT-style).

    Components:
        - Embedding layer: Maps token IDs -> dense vectors of size (embed_dim).
        - Transformer layers: A stack of N Transformer blocks (each with attention + feedforward).
        - RMSNorm: Normalization before the output projection.
        - Linear projection: Maps hidden states -> vocabulary logits.

    Args:
        args (ModelArgs):
            - vocab_size (int): Size of the tokenizer vocabulary.
            - embed_dim (int): Dimension of token embeddings and hidden states.
            - n_layers (int): Number of Transformer layers.
            - dtype (torch.dtype): Computation precision (e.g., torch.bfloat16).
    """
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.embed_dim, dtype=args.dtype)
        self.layers = nn.ModuleList(
            [Transformer(args) for i in range(args.n_layers)]
        )
        self.logits = nn.Linear(args.embed_dim, args.vocab_size, dtype=args.dtype)
        self.norm = RMSNorm()
    
    def forward(self, input_ids):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.LongTensor):
                Shape: [batch, seq_len]
                Each value is a token index from the tokenizer's vocabulary.

        Returns:
            torch.FloatTensor:
                Shape: [batch, seq_len, vocab_size]
                Unnormalized logits for each token position across the vocabulary.

        Steps:
            1. Convert input_ids to embeddings. -> [B, L, D]
            2. Pass embeddings sequentially through all Transformer layers.
            3. Apply RMSNorm to stabilize training.
            4. Project normalized hidden states to vocabulary logits.
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        return self.logits(x)
    
args = ModelArgs()

model = Model(args=args)


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token

text = "Hello, this is a test."
input_ids = tokenizer(text, return_tensors="pt").input_ids  

