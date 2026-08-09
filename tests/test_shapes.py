import os 
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np 
import jax.numpy as jnp
from flax import nnx
from config import Config
import data
from model import (
    PositionalEmbedding,
    TokenEmbedding,
    SelfAttentionHead,
    MultiHeadAttention,
    FeedForward,
    Block,
    DigitTransformer
)

def small_config(**overrides):
    cfg=Config(n_digits=2, n_embd=16, n_head=2, n_layer=2)
    for k,v in overrides.items():
        setattr(cfg,k,v)
    return cfg

def test_embedding_shape():
    cfg= small_config()
    emb= TokenEmbedding(cfg.vocab_size, cfg.n_embd, rngs= nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size), dtype= jnp.int32)
    out= emb(x)
    assert out.shape== (4, cfg.block_size, cfg.n_embd)

def test_positional_embedding_shape():
    cfg=small_config()
    pos= PositionalEmbedding(cfg.block_size,cfg.n_embd, rngs=nnx.Rngs(cfg.seed))
    out = pos(cfg.block_size)
    assert out.shape ==(cfg.block_size,cfg.n_embd)

def test_attention_head_shape():
    cfg=small_config()
    head_size= cfg.n_embd//cfg.n_head
    head= SelfAttentionHead(cfg.n_embd,head_size,cfg.block_size,cfg.dropout,rngs=nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size,cfg.n_embd))
    out=head(x,training=False)
    assert out.shape==(4,cfg.block_size,head_size)

def test_multi_head_attention_shape():
    cfg=small_config()
    mha= MultiHeadAttention(cfg.n_embd,cfg.n_head,cfg.block_size,cfg.dropout,rngs=nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size,cfg.n_embd))
    out=mha(x,training=False)
    assert out.shape==(4,cfg.block_size,cfg.n_embd)

def test_feedforward_shape():
    cfg = small_config()
    ff = FeedForward(cfg.n_embd, cfg.dropout, rngs=nnx.Rngs(cfg.seed))
    x = jnp.zeros((4, cfg.block_size, cfg.n_embd))
    out = ff(x, training=False)
    assert out.shape == (4, cfg.block_size, cfg.n_embd)

def test_block_shape():
    cfg= small_config()
    block= Block(cfg.n_embd,cfg.n_head,cfg.block_size,cfg.dropout, rngs=nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size,cfg.n_embd))
    out=block(x,training=False)
    assert out.shape==(4,cfg.block_size,cfg.n_embd)

def test_digit_transformer_forward_pass_shape():
    cfg=small_config()
    model= DigitTransformer(cfg, rngs= nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size), dtype= jnp.int32)
    out=model(x,training=False)
    assert out.shape==(4,cfg.block_size,cfg.vocab_size)

def test_digit_transformer_is_seeded_deterministically():
    cfg=small_config()
    m1=DigitTransformer(cfg, rngs= nnx.Rngs(cfg.seed))
    m2=DigitTransformer(cfg, rngs= nnx.Rngs(cfg.seed))
    x= jnp.zeros((4,cfg.block_size), dtype= jnp.int32)
    out1=m1(x,training=False)
    out2=m2(x,training=False)
    assert jnp.allclose(out1,out2)

def test_encode_decode_rountrip():
    cfg= small_config()
    rng= np.random.default_rng(cfg.seed)
    ex= data.make_example(rng, cfg.n_digits)
    assert ex.shape == (data.seq_len(cfg.n_digits),)
    decoded= data.decode(ex.tolist())
    assert len(decoded)==data.seq_len(cfg.n_digits)
    assert all(c in data.VOCAB for c in decoded)

def test_equationformat_is_correct():
    cfg= small_config()
    a,b=85,50
    s= f"{a:>{2}}+{b:>{2}}={a+b:>{3}}"
    assert s== "85+50=135"
    assert len(data.encode(s))==data.seq_len(2)
    assert data.decode(data.encode(s))==s

def test_loss_marks_covers_exactly_sum_digits():
    cfg= small_config()
    mask= data.loss_mask(cfg.n_digits,cfg.block_size)
    assert mask.sum()== cfg.n_digits+1
    assert mask.dtype== bool

def test_data_batch_shapes():
    rng= np.random.default_rng(123)
    cfg= small_config()
    x,y = data.get_batch(rng,cfg.n_digits,batch_size=8)
    assert x.shape==(8,cfg.block_size)
    assert y.shape==(8,cfg.block_size)




