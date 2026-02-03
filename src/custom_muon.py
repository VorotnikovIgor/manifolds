import math
import torch
from msign import msign

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
    return A.T if should_tranpose else A
    # # Descend on the primal problem
    # new_W = W - eta * A
    # # Retract to the manifold
    # new_W = msign(new_W)
    # # Restore the shape of the solution and return
    # return new_W.T if should_tranpose else new_W

def newtonschulz5_autograd_safe(A, steps=5, eps=1e-7, wo_normalization=False):
    """
    Newton-Schulz итерация БЕЗ in-place операций.
    Полностью совместима с PyTorch autograd.
    """
    assert A.ndim == 2

    # Константы алгоритма
    a, b, c = (2, -1.5, 0.5)

    # 1. ВСЕГДА создаем новую переменную, никогда не модифицируем вход
    X = A.clone() if A.requires_grad else A
    #X = A

    # 2. Избегаем in-place операций: используем X = X / norm вместо X /= norm
    if not wo_normalization:
        norm_val = torch.linalg.norm(X)
        X = X / (norm_val + eps)

    # 3. Сохраняем флаг транспонирования
    original_shape = X.shape
    if A.size(0) > A.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    # 4. Итерации Newton-Schulz - каждая создает новый тензор
    for i in range(steps):
        # Все промежуточные переменные создаются заново
        C = X @ X.T
        C_sq = C @ C  # Явно вычисляем квадрат
        B = b * C + c * C_sq

        # КРИТИЧЕСКИ ВАЖНО: создаем новый тензор
        X_new = a * X + B @ X
        X = X_new  # Присваивание, но не in-place модификация!

    # 5. Возвращаем к исходной форме
    if transposed:
        X = X.T

    # Проверяем размеры
    assert X.shape == original_shape, f"Shape mismatch: {X.shape} != {original_shape}"

    return X

def power(A, pow=5):
    X = A.clone() if A.requires_grad else A

    if A.size(0) > A.size(1):
        X = X.T
        transposed = True
    else:
        transposed = False

    C = X @ X.T
    for i in range(pow - 1):
        C = C @ C

    res = C @ X

    if transposed:
        X = X.T

    return res

def make_orthogonal(grads):
    l = len(grads)
    for i in range(1, l):
        g = grads[i]
        for j in range(i):
            if torch.norm(grads[i]) < 1e-7:
                continue
            grads[j] = grads[j] - (torch.trace(grads[j] @ grads[i].T) / torch.norm(grads[i]) ** 2) * grads[i]

    return grads

def distance_from_tangent(w, g):
    m = w.T @ g + g.T @ w
    return torch.norm(m) ** 2 / 4, w @ m

def custom_muon(W, G, eta=0.1, T=100, mode='ns', start='warm', params={'steps' : 4}):
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T

    # print(W.shape)
    n, k = W.shape
    if start == 'basic':
        A = torch.nn.Parameter(torch.randn(n, k) / torch.math.sqrt(n * k) * (1.4 if mode == 'ns' else 1))
    else:
        A = torch.nn.Parameter(manifold_muon(W, G, steps=1, alpha=0.1) * (1.4 if mode == 'ns' else 1))
    # opt = -torch.sum(torch.svd(G)[1])
    # losses = []
    # times = []

    # start = time.time()
    # As = []
    for epoch in range(T):
        A.grad = None
        with torch.enable_grad():
            if mode == 'ns':
                output_norm = torch.norm(newtonschulz5_autograd_safe(A, steps=params['steps'], wo_normalization=True))
                relu = torch.nn.ReLU()
                loss_norm = relu(output_norm - torch.math.sqrt(k))
            else:
                output_norm = torch.norm(power(A, pow=params['pow']))
                eps = 0.1 * k
                relu = torch.nn.ReLU()
                loss_norm = relu(output_norm - torch.math.sqrt(eps))
            # print("A.requires_grad:", A.requires_grad)
            # print("output_norm.requires_grad:", output_norm.requires_grad)
            # print("loss_norm.requires_grad:", loss_norm.requires_grad)
            # print("output_norm.grad_fn:", output_norm.grad_fn)
            # print("loss_norm.grad_fn:", loss_norm.grad_fn)
            loss_norm.backward()
            grad_norm = A.grad.clone()

        A.grad = None
        loss_tang, grad_tang = distance_from_tangent(W, A)
        #loss_tang.backward()
        #grad_tang = A.grad.clone()

        A.grad = None
        # loss_f = - torch.trace(G.T @ A)
        #loss_f.backward()
        grad_f = -G #A.grad.clone()

        grad_f, grad_tang, grad_norm = make_orthogonal([grad_f, grad_tang, grad_norm])
        # A.grad = grad_f.clone()
        # optimizer.step()
        with torch.no_grad():
            coef = (0.0005 if n > 500 else 0.005)
            A.data = A.data - coef * (0.1 * grad_f + grad_norm + 10 * grad_tang) # * (1 - epoch / T)
            # print("changed")
        #print(A)
        # 6. Обновляем параметры
        #optimizer.step()
        #print(A)
        #losses.append(loss.item())
        if torch.any(torch.isnan(grad_norm)):
            print(epoch)
            print(f"grad_norm = {grad_norm}")

        if torch.any(torch.isnan(grad_f)):
            print(f"grad_f = {grad_f}")

        # if epoch == 0 or (epoch + 1) % 100 == 0:
        #     print(f"Epoch {epoch + 1}: loss_norm = {loss_norm.item():5f}, |A|_2 = {torch.svd(A)[1][0]:.3f}, loss_tang = {loss_tang:5f}, loss_f = {loss_f:.3f}, opt = {opt * torch.svd(A)[1][0]:.3f}")
            #print(torch.svd(A)[1])
        # new_start = time.time()
        # times.append(time.time() - start)
        # start = new_start
        # As.append(A.clone())
    # print(f"Time = {sum(times):.0f} sec")
    with torch.no_grad():
        if mode == 'ns':
            A.data /= 1.4
    new_W = W - eta * A
    new_W = msign(new_W)
    return new_W.T if should_tranpose else new_W