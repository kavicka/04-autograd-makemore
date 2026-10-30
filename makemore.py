"""Character-level language models that invent Slovak municipality names.

baseline 0: uniform guess over the alphabet
baseline 1: bigram counts (next char depends only on the current char)
model:      MLP over the last 3 characters (Bengio et al. 2003), in PyTorch
"""

import json
import math
import random

import torch
import torch.nn.functional as F

BLOCK = 3  # context length: how many previous characters the MLP sees
N_EMBD = 10  # size of each character's embedding vector
N_HIDDEN = 200
STEPS = 30_000
BATCH = 32
SEED = 42
WEIGHT_DECAY = 1e-3  # L2 penalty on W1, W2: the sweep showed it shrinks the train/val gap

words = [w for w in open("data/obce.txt", encoding="utf-8").read().splitlines() if w]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0  # one special token marks both start and end of a name
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

random.seed(SEED)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
splits = {"train": words[:n1], "val": words[n1:n2], "test": words[n2:]}


# --- bigram baseline ---------------------------------------------------------------
def bigram_model(train_words: list[str], smoothing: float = 1.0) -> torch.Tensor:
    N = torch.zeros((V, V))
    for w in train_words:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1
    P = (N + smoothing) / (N + smoothing).sum(1, keepdim=True)  # each row: P(next | current)
    return N, P


def bigram_nll(P: torch.Tensor, ws: list[str]) -> float:
    log_likelihood, n = 0.0, 0
    for w in ws:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            log_likelihood += torch.log(P[stoi[a], stoi[b]]).item()
            n += 1
    return -log_likelihood / n  # average negative log-likelihood per character


def bigram_sample(P: torch.Tensor, g: torch.Generator) -> str:
    out, ix = [], 0
    while True:
        ix = torch.multinomial(P[ix], 1, generator=g).item()
        if ix == 0:
            return "".join(out)
        out.append(itos[ix])


# --- MLP ---------------------------------------------------------------------
def build_dataset(ws: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []
    for w in ws:
        context = [0] * BLOCK
        for ch in w + ".":
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]  # slide the window
    return torch.tensor(X), torch.tensor(Y)


def init_params(g: torch.Generator) -> list[torch.Tensor]:
    C = torch.randn((V, N_EMBD), generator=g)  # embedding table: one vector per character
    W1 = torch.randn((BLOCK * N_EMBD, N_HIDDEN), generator=g) * (5 / 3) / (BLOCK * N_EMBD) ** 0.5  # Kaiming init for tanh
    b1 = torch.randn(N_HIDDEN, generator=g) * 0.01
    W2 = torch.randn((N_HIDDEN, V), generator=g) * 0.01  # small → near-uniform predictions at start
    b2 = torch.zeros(V)
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


def forward(params: list[torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    C, W1, b1, W2, b2 = params
    emb = C[X]  # (B, 3, 10)
    h = torch.tanh(emb.view(-1, BLOCK * N_EMBD) @ W1 + b1)  # (B, 30) @ (30, 200) → (B, 200)
    return h @ W2 + b2  # (B, V) logits


@torch.no_grad()
def split_loss(params, X, Y) -> float:
    return F.cross_entropy(forward(params, X), Y).item()


def mlp_sample(params, g: torch.Generator, temperature: float = 1.0) -> str:
    out, context = [], [0] * BLOCK
    while True:
        logits = forward(params, torch.tensor([context]))
        probs = F.softmax(logits / temperature, dim=1)
        ix = torch.multinomial(probs, 1, generator=g).item()
        if ix == 0:
            return "".join(out)
        out.append(itos[ix])
        context = context[1:] + [ix]


def overfit_one_batch(data) -> list[float]:
    """Debugging habit: a correct model must be able to memorise a single batch (loss → ~0)."""
    g = torch.Generator().manual_seed(SEED)
    params = init_params(g)
    Xb, Yb = data["train"][0][:BATCH], data["train"][1][:BATCH]
    losses = []
    for _ in range(1000):
        loss = F.cross_entropy(forward(params, Xb), Yb)
        for p in params:
            p.grad = None
        loss.backward()
        for p in params:
            p.data += -0.1 * p.grad
        losses.append(loss.item())
    return losses


def main() -> None:
    print(f"{len(words)} names, vocabulary {V} characters (incl. '.'), split {[len(s) for s in splits.values()]}")
    uniform = math.log(V)

    N, P = bigram_model(splits["train"])
    bigram = {k: bigram_nll(P, ws) for k, ws in splits.items()}

    data = {k: build_dataset(ws) for k, ws in splits.items()}
    print(f"MLP examples: { {k: len(v[1]) for k, v in data.items()} }")

    obo = overfit_one_batch(data)
    print(f"overfit one batch: loss {obo[0]:.3f} → {obo[-1]:.3f}")

    g = torch.Generator().manual_seed(SEED)
    params = init_params(g)
    print(f"MLP parameters: {sum(p.nelement() for p in params):,}")
    Xtr, Ytr = data["train"]
    curve = []
    for i in range(STEPS):
        ix = torch.randint(0, Xtr.shape[0], (BATCH,), generator=g)
        loss = F.cross_entropy(forward(params, Xtr[ix]), Ytr[ix])
        reg = WEIGHT_DECAY * ((params[1] ** 2).sum() + (params[3] ** 2).sum())
        for p in params:
            p.grad = None
        (loss + reg).backward()
        lr = 0.1 if i < STEPS * 2 // 3 else 0.01  # step decay
        for p in params:
            p.data += -lr * p.grad
        if i % 500 == 0 or i == STEPS - 1:
            tr, va = split_loss(params, *data["train"]), split_loss(params, *data["val"])
            curve.append({"step": i, "batch": loss.item(), "train": tr, "val": va})
            if i % 5000 == 0:
                print(f"step {i:6d}  batch {loss.item():.3f}  train {tr:.3f}  val {va:.3f}")
    mlp = {k: split_loss(params, *d) for k, d in data.items()}

    print("\nNLL per character (lower is better)")
    print(f"  {'model':<22} {'train':>7} {'val':>7} {'test':>7}")
    print(f"  {'uniform guess':<22} {uniform:7.3f} {uniform:7.3f} {uniform:7.3f}")
    print(f"  {'bigram (counts)':<22} {bigram['train']:7.3f} {bigram['val']:7.3f} {bigram['test']:7.3f}")
    print(f"  {'MLP (3 chars context)':<22} {mlp['train']:7.3f} {mlp['val']:7.3f} {mlp['test']:7.3f}")

    gs = torch.Generator().manual_seed(2026)
    train_set = set(splits["train"])
    samples_bigram = [bigram_sample(P, gs) for _ in range(20)]
    samples_mlp = [mlp_sample(params, gs) for _ in range(20)]
    samples_cold = [mlp_sample(params, gs, temperature=0.7) for _ in range(20)]
    many = [mlp_sample(params, gs) for _ in range(500)]
    novel = sum(s not in set(words) for s in many) / len(many)
    in_val_test = sum(s in set(splits["val"] + splits["test"]) for s in many)
    print(f"\nbigram samples: {', '.join(samples_bigram)}")
    print(f"MLP samples:    {', '.join(samples_mlp)}")
    print(f"MLP, T=0.7:     {', '.join(samples_cold)}")
    print(f"\nof 500 MLP samples: {novel:.0%} are new names, {in_val_test} are real municipalities the model never saw in training")

    with open("results_makemore.json", "w") as f:
        json.dump({
            "n_words": len(words), "vocab": V, "chars": ["."] + chars, "splits": {k: len(v) for k, v in splits.items()},
            "uniform": uniform, "bigram": bigram, "mlp": mlp, "curve": curve, "overfit_one_batch": obo,
            "bigram_counts": N.tolist(), "embeddings": params[0].detach().tolist(),
            "samples_bigram": samples_bigram, "samples_mlp": samples_mlp, "samples_mlp_t07": samples_cold, "novel_fraction": novel,
            "hits_in_val_test": in_val_test, "n_params": sum(p.nelement() for p in params),
            "real_hits": sorted({s for s in many if s in set(splits['val'] + splits['test'])}),
        }, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
