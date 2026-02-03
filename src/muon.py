import math
import torch
from msign import msign

@torch.no_grad()
def muon(W, G, eta=0.1, on_manifold=True):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    A = msign(G)
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    if on_manifold:
        new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W