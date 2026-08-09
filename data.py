import numpy as np

VOCAB = "0123456789 +="
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# keep lowercase alias used by config.vocab_size = len(data.vocab)
vocab = VOCAB


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


def make_example(rng: np.random.Generator, n_digits: int) -> np.ndarray:
    max_val = 10**n_digits
    a = int(rng.integers(0, max_val))
    b = int(rng.integers(0, max_val))
    c = a + b
    s = f"{a:>{n_digits}}+{b:>{n_digits}}={c:>{n_digits+1}}"
    return np.array(encode(s), dtype=np.int32)


def seq_len(n_digits: int) -> int:
    return 3 * n_digits + 3


def get_batch(rng: np.random.Generator, n_digits: int, batch_size: int):
    seqs = np.stack([make_example(rng, n_digits) for _ in range(batch_size)])
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    return x, y


def loss_mask(n_digits: int, block_size: int) -> np.ndarray:
    mask = np.zeros(block_size, dtype=bool)
    mask[2 * n_digits + 1 :] = True
    return mask
