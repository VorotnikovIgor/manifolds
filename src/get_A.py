import math
import torch
from msign import msign
import time

@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    # Ascend on the dual problem to find the update direction A
    times = []
    As = []
    start = time.time()
    for step in range(steps):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
        # new_start = time.time()
        times.append(time.time() - start)
        # start = new_start
        As.append(A.clone())
    return As, times
