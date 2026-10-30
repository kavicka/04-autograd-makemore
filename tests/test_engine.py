"""Our gradients must match PyTorch's (torch.allclose) on every operation and on a whole MLP."""

import random

import torch

from micrograd.engine import Value
from micrograd.nn import MLP


def test_each_operation():
    ops = {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "sub": lambda a, b: a - b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: a**3 + b**-2,
        "exp": lambda a, b: (a * 0.5).exp() + b,
        "log": lambda a, b: (a * a + 1).log() * b,
        "tanh": lambda a, b: (a * b).tanh(),
        "relu": lambda a, b: (a - b).relu() + (b - a).relu() * 2,
    }
    for name, f in ops.items():
        a, b = Value(1.5), Value(-0.7)
        out = f(a, b)
        out.backward()

        ta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        tb = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
        tf = {"tanh": lambda a, b: torch.tanh(a * b), "exp": lambda a, b: torch.exp(a * 0.5) + b,
              "log": lambda a, b: torch.log(a * a + 1) * b,
              "relu": lambda a, b: torch.relu(a - b) + torch.relu(b - a) * 2}.get(name, f)
        tout = tf(ta, tb)
        tout.backward()

        assert torch.allclose(torch.tensor(out.data, dtype=torch.float64), tout.detach()), name
        assert torch.allclose(torch.tensor([a.grad, b.grad], dtype=torch.float64), torch.stack([ta.grad, tb.grad])), name


def test_shared_node_accumulates_gradient():
    # x is used twice: its gradient must be the SUM of both paths (that's why backward uses +=)
    x = Value(3.0)
    y = x * x + x  # dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == 7.0


def test_karpathy_expression():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()

    ta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tc = ta + tb
    td = ta * tb + tb**3
    tc = tc + tc + 1
    tc = tc + 1 + tc + (-ta)
    td = td + td * 2 + (tb + ta).relu()
    td = td + 3 * td + (tb - ta).relu()
    te = tc - td
    tf = te**2
    tg = tf / 2.0
    tg = tg + 10.0 / tf
    tg.backward()

    assert abs(g.data - tg.item()) < 1e-9
    assert torch.allclose(torch.tensor([a.grad, b.grad], dtype=torch.float64), torch.stack([ta.grad, tb.grad]))


def test_mlp_matches_torch():
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1.0]
    out = model([Value(v) for v in x])
    out.backward()

    # rebuild the same network in torch with the same weights
    tx = torch.tensor(x, dtype=torch.float64)
    params = []
    h = tx
    for i, layer in enumerate(model.layers):
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        params.append((W, b))
        h = W @ h + b
        if i != len(model.layers) - 1:
            h = torch.tanh(h)
    h.sum().backward()

    assert abs(out.data - h.item()) < 1e-9
    for layer, (W, b) in zip(model.layers, params):
        ours_W = torch.tensor([[w.grad for w in n.w] for n in layer.neurons], dtype=torch.float64)
        ours_b = torch.tensor([n.b.grad for n in layer.neurons], dtype=torch.float64)
        assert torch.allclose(ours_W, W.grad)
        assert torch.allclose(ours_b, b.grad)
