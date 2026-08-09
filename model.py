from flax import nnx
import jax
import jax.numpy as jnp

from config import Config

class TokenEmbedding(nnx.Module):
    def __init__(self, vocab_size: int, n_embd: int, *, rngs: nnx.Rngs):
        self.embed= nnx.Embed(vocab_size, n_embd, rngs=rngs)
    def __call__(self,idx):
        return self.embed(idx)

class PositionalEmbedding(nnx.Module):
    def __init__(self, block_size: int, n_embd: int,*, rngs: nnx.Rngs):
        self.embed =nnx.Embed(block_size, n_embd, rngs=rngs)
    def __call__(self,T:int):
        return self.embed(jnp.arange(T))

class SelfAttentionHead(nnx.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float, *, rngs: nnx.Rngs):
        self.key = nnx.Linear(n_embd, head_size, use_bias=False, rngs= rngs)
        self.query = nnx.Linear(n_embd, head_size, use_bias=False, rngs= rngs)
        self.value = nnx.Linear(n_embd, head_size, use_bias=False, rngs= rngs)
        self.dropout= nnx.Dropout(dropout, rngs=rngs)
        self.causal_mask= jnp.tril(jnp.ones((block_size,block_size),dtype= bool))
    def __call__(self, x, training: bool):
        B,T,C =x.shape
        k= self.key(x)
        q= self.query(x)
        wei= q @ jnp.swapaxes(k, -2,-1) * (k.shape[-1]**-0.5)
        wei = jnp.where(self.causal_mask[:T,:T], wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, deterministic=not training)
        v= self.value(x)
        return wei @ v

class MultiHeadAttention(nnx.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, *, rngs: nnx.Rngs):
        head_size= n_embd // n_head
        self.heads= nnx.List([
            SelfAttentionHead(n_embd, head_size, block_size, dropout, rngs=rngs) 
            for _ in range(n_head)])
        self.proj = nnx.Linear(n_embd, n_embd, rngs=rngs)
        self.dropout= nnx.Dropout(dropout, rngs=rngs)
    def __call__(self, x, training: bool):
        out= jnp.concatenate([h(x,training) for h in self.heads], axis=-1)
        return self.dropout(self.proj(out), deterministic=not training)

class FeedForward(nnx.Module):
    def __init__(self,n_embd: int, dropout:float, *, rngs:nnx.Rngs):
        self.fc1= nnx.Linear(n_embd, 4*n_embd, rngs=rngs)
        self.fc2= nnx.Linear(4*n_embd, n_embd,rngs=rngs)
        self.dropout= nnx.Dropout(dropout, rngs=rngs)
    def __call__(self, x, training: bool):
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x, deterministic=not training)

class Block(nnx.Module):
    def __init__(self, n_embd: int,n_head: int, block_size: int,dropout: float, *, rngs:nnx.Rngs):
        self.sa= MultiHeadAttention(n_embd, n_head, block_size, dropout, rngs=rngs)
        self.ffwd= FeedForward(n_embd, dropout, rngs=rngs)
        self.ln1= nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2= nnx.LayerNorm(n_embd, rngs=rngs)

    def __call__(self, x, training: bool):
        x= x+ self.sa(self.ln1(x),training)
        x= x+ self.ffwd(self.ln2(x), training)
        return x

class DigitTransformer(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        self.TokenEmbedding= TokenEmbedding(cfg.vocab_size, cfg.n_embd, rngs=rngs)
        self.PositionalEmbedding= PositionalEmbedding(cfg.block_size, cfg.n_embd, rngs=rngs)
        self.blocks= nnx.List([Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout, rngs=rngs) for _ in range(cfg.n_layer)])
        self.ln_f= nnx.LayerNorm(cfg.n_embd, rngs=rngs)
        self.lm_head= nnx.Linear(cfg.n_embd, cfg.vocab_size, rngs=rngs)

    def __call__(self, idx, training:bool=True):
        B,T= idx.shape
        x= self.TokenEmbedding(idx) + self.PositionalEmbedding(T)
        for block in self.blocks:
            x= block(x,training)
        x= self.ln_f(x)
        return self.lm_head(x)
