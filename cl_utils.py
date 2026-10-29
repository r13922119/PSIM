"""cl_utils.py — the two forms of Cl (dropout on the frozen W_pre / W_pre,K output of a query/value module).
Used by variant_finetune.py (training) and p0_eval.py (step-0 damage), so both apply exactly the same dropout.

  coord  (RoRA's main text, what we have used so far): inverted dropout on every output coordinate of y = W x + b
         y' = y * m / (1 - p),  m_j ~ Bernoulli(1 - p), one m per token per coordinate (1024 per token).
  svd    (your 10/19 (d)): inverted dropout on the components of W x along W's own left singular directions.
         W = U S V^T (rank k: the k directions W_pre,K keeps; k = 1024 for an untruncated W). With z = W x = y - b and
         c = U^T z (k numbers per token): c' = c * m / (1 - p), one m per token per direction; y' = y + U (c' - c).
         The bias b is not dropped (it is not a direction of W). Nothing outside span(U) changes; for a rank-k W, z lies
         in span(U), so this is exactly "drop rank-one pieces s_i u_i v_i^T x".
Both are unbiased (E[y'] = y), and both inject the same expected relative noise energy into W x:
  E||z' - z||^2 / ||z||^2 = p / (1 - p)   (coord: per coordinate; svd: per direction, U orthonormal)
except that coord also drops the bias coordinates. So equal p means equal noise energy on W x, not equal damage to the
model; damage is measured separately (p0_eval.py, step-0 AUC).
"""
import torch
import torch.nn.functional as F

FORMS = ("coord", "svd")


def left_basis(W, k=None):
    """the first k left singular vectors of W (float32, columns orthonormal), from torch.linalg.svd on float32 —
    the same call variant_finetune.py uses for Tr's U, so for a W_pre,K the two use the same directions"""
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    k = min(W.shape) if k is None else int(k)
    return U[:, :k].contiguous()


def svd_dropout(y, bias, U, p, training=True):
    """y: (..., d_out) output of the frozen Linear (W x + b); U: (d_out, k). Returns y with its W-direction components dropped."""
    if not training or p == 0:
        return y
    z = y - bias if bias is not None else y
    c = z @ U                                     # (..., k)
    return y + (F.dropout(c, p=p, training=True) - c) @ U.T


def cl_apply(y, module, form, p, U=None):
    """the Cl dropout for one frozen Linear `module` (its bias is used by the svd form)"""
    if form == "coord":
        return F.dropout(y, p=p, training=module.training)
    if form == "svd":
        return svd_dropout(y, module.bias, U, p, training=module.training)
    raise ValueError(form)