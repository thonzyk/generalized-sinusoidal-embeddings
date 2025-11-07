import numpy as np
import math
import math
from typing import Sequence, Union



def positional_encoding_auto(
    seq_len: int,
    d_model: int,
) -> np.ndarray:
    """
    Compute a 1-D sinusoidal positional encoding with automatically scaled frequencies.

    The last sin/cos pair spans a half-period over the full sequence:
    pos=0 → cos(0)=1, pos=seq_len-1 → cos(π)=-1.
    Frequencies for earlier pairs are spaced geometrically to satisfy this constraint.

    Parameters
    ----------
    seq_len : int
        Sequence length (number of positions). Must be >= 1.
    d_model : int
        Embedding size. Must be a positive even integer due to sin/cos pairing.

    Returns
    -------
    np.ndarray
        Array of shape (seq_len, d_model), dtype float32.
        Even channels are sin, odd channels are cos.

    Examples
    --------
    >>> pe = positional_encoding_auto(seq_len=8, d_model=6)
    >>> pe.shape
    (8, 6)
    >>> # Last pair's cosine goes from +1 to -1
    >>> np.allclose(pe[0, 1], 1.0) and np.allclose(pe[-1, 1], -1.0)
    True
    """
    if not isinstance(seq_len, int) or seq_len < 1:
        raise ValueError("seq_len must be a positive integer.")
    if not isinstance(d_model, int) or d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even integer.")
    if seq_len == 1:
        return np.zeros((1, d_model), dtype=np.float32)

    Lm1 = seq_len - 1
    delta = math.pi  # fixed: final_diff_limit == 2.0

    # Handle the degenerate exponent when d_model == 2
    if d_model == 2:
        angle_rates = np.full(2, delta / Lm1, dtype=np.float64)
    else:
        expo = 1.0 - 2.0 / d_model                 # = (d_model-2)/d_model
        auto_rate = (delta / Lm1) ** (-1.0 / expo) # solve last-pair constraint
        i = np.arange(d_model, dtype=np.float64)
        pair_j = np.floor(i / 2.0)                 # 0..P-1
        angle_rates = auto_rate ** (-(2.0 * pair_j) / d_model)

    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    angles = positions * angle_rates[None, :]

    out = np.empty_like(angles)
    out[:, 0::2] = np.sin(angles[:, 0::2])
    out[:, 1::2] = np.cos(angles[:, 1::2])
    return out.astype(np.float32, copy=False)


def positional_encoding_auto_ND(
    seq_length: Union[int, Sequence[int]],
    emb_dim: int,
    space_dim: int,
) -> np.ndarray:
    """
    Build an N-D positional embedding by concatenating per-axis sinusoidal encodings.

    Each spatial axis receives a sub-embedding of size `emb_dim // space_dim`.
    The final tensor is the concatenation across axes, with an added batch dim.

    Parameters
    ----------
    seq_length : int or Sequence[int]
        If int, the same length is used for all `space_dim` axes.
        If a sequence, must have length == `space_dim` and all items positive ints.
    emb_dim : int
        Total embedding size across all axes. Must be divisible by `space_dim`
        and result in an even sub-dimension for sin/cos pairing.
    space_dim : int
        Number of spatial axes, e.g., 2 for images, 3 for volumes.

    Returns
    -------
    np.ndarray
        Shape (1, *seq_lengths, emb_dim), dtype float32.

    Examples
    --------
    >>> pe = positional_encoding_auto_ND((32, 64), emb_dim=128, space_dim=2)
    >>> pe.shape
    (1, 32, 64, 128)
    """
    # Normalize seq lengths
    if not isinstance(space_dim, int) or space_dim <= 0:
        raise ValueError("space_dim must be a positive integer.")

    if isinstance(seq_length, int):
        if seq_length < 1:
            raise ValueError("seq_length must be >= 1.")
        seqs = [int(seq_length)] * space_dim
    else:
        if not isinstance(seq_length, (list, tuple)):
            raise TypeError("seq_length must be int or a sequence of ints.")
        if len(seq_length) != space_dim:
            raise ValueError("len(seq_length) must equal space_dim.")
        if not all(isinstance(x, int) and x >= 1 for x in seq_length):
            raise ValueError("All elements of seq_length must be positive ints.")
        seqs = list(seq_length)

    # Validate dimensions
    if not isinstance(emb_dim, int) or emb_dim <= 0:
        raise ValueError("emb_dim must be a positive integer.")
    if emb_dim % space_dim != 0:
        raise ValueError("emb_dim must be divisible by space_dim.")
    sub_dim = emb_dim // space_dim
    if sub_dim % 2 != 0:
        raise ValueError("emb_dim // space_dim must be even (sin/cos pairs).")

    # Build per-axis encodings and broadcast to the full grid
    broadcast_shape = tuple(seqs) + (sub_dim,)
    per_axis = []
    for axis, L in enumerate(seqs):
        # 1-D encoding for this axis, shape (L, sub_dim)
        pe_axis = positional_encoding_auto(seq_len=L, d_model=sub_dim)

        # Reshape to broadcast over all other axes: [1,..., L(at axis), ...,1, sub_dim]
        shape = [1] * space_dim + [sub_dim]
        shape[axis] = L
        pe_axis = pe_axis.reshape(tuple(shape))

        # Broadcast to (L1, L2, ..., Ln, sub_dim)
        pe_axis = np.broadcast_to(pe_axis, broadcast_shape)
        per_axis.append(pe_axis)

    # Concatenate axis encodings along the last dim → (*seqs, emb_dim)
    emb = np.concatenate(per_axis, axis=-1)

    # Add batch dimension at front for consistency with common frameworks
    return emb[None, ...].astype(np.float32, copy=False)


def positional_encoding_vanilla(seq_length: int, emb_dim: int) -> np.ndarray:
    """
    Classical sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        seq_len: number of positions (>=1).
        d_model: embedding dimension (even).

    Returns:
        np.ndarray of shape (seq_len, d_model), dtype float32.
        Even dims use sin, odd dims use cos.
    """
    if not isinstance(seq_length, int) or seq_length < 1:
        raise ValueError("seq_len must be a positive integer.")
    if not isinstance(emb_dim, int) or emb_dim <= 0 or emb_dim % 2 != 0:
        raise ValueError("d_model must be a positive even integer.")

    # positions: (seq_len, 1)
    positions = np.arange(seq_length, dtype=np.float64)[:, None]
    # pair indices: (d_model/2,)
    i = np.arange(emb_dim // 2, dtype=np.float64)

    # angle rates: 1 / 10000^{2i/d_model}
    angle_rates = 1.0 / np.power(10000.0, (2.0 * i) / emb_dim)  # (d_model/2,)

    # angles: (seq_len, d_model/2)
    angles = positions * angle_rates[None, :]

    pe = np.empty((seq_length, emb_dim), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles)  # even channels
    pe[:, 1::2] = np.cos(angles)  # odd channels
    return pe.astype(np.float32, copy=False)


if __name__ == "__main__":
    # Quick sanity checks
    pe1 = positional_encoding_auto(seq_len=8, d_model=6)
    print("1D:", pe1.shape, pe1.dtype)

    pe2 = positional_encoding_auto_ND((4, 5), emb_dim=16, space_dim=2)
    print("2D:", pe2.shape, pe2.dtype)
