"""A tiny neural-network library on top of Value: Neuron → Layer → MLP."""

import random

from micrograd.engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, n_in: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # w·x + b
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x: list[Value]) -> list[Value] | Value:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in: int, n_outs: list[int]):
        sizes = [n_in] + n_outs
        # every layer has tanh except the last one, which outputs a raw score
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=i != len(n_outs) - 1) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
