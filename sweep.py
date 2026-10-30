"""Small hyperparameter sweep for the makemore MLP: which change actually lowers validation loss?"""

import json

import torch
import torch.nn.functional as F

import makemore as m

CONFIGS = [  # (hidden, embedding, weight decay, steps, batch)
    (64, 8, 0.0, 20_000, 32),
    (100, 10, 0.0, 20_000, 32),
    (200, 10, 0.0, 30_000, 32),
    (200, 10, 0.0, 20_000, 128),
    (100, 10, 1e-3, 20_000, 64),
    (200, 10, 1e-3, 30_000, 32),
]


def run(hidden, embd, wd, steps, batch, data) -> dict:
    m.N_HIDDEN, m.N_EMBD = hidden, embd
    g = torch.Generator().manual_seed(m.SEED)
    params = m.init_params(g)
    Xtr, Ytr = data["train"]
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g)
        loss = F.cross_entropy(m.forward(params, Xtr[ix]), Ytr[ix]) + wd * ((params[1] ** 2).sum() + (params[3] ** 2).sum())
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < steps * 2 // 3 else 0.01
        for p in params:
            p.data += -lr * p.grad
    tr, va = m.split_loss(params, *data["train"]), m.split_loss(params, *data["val"])
    return {"hidden": hidden, "embd": embd, "weight_decay": wd, "steps": steps, "batch": batch,
            "params": sum(p.nelement() for p in params), "train": tr, "val": va, "gap": va - tr}


def main() -> None:
    data = {k: m.build_dataset(ws) for k, ws in m.splits.items()}
    rows = []
    print(f"{'hidden':>6} {'embd':>4} {'wd':>6} {'steps':>6} {'batch':>5} {'params':>7} {'train':>6} {'val':>6} {'gap':>6}")
    for cfg in CONFIGS:
        r = run(*cfg, data)
        rows.append(r)
        print(f"{r['hidden']:>6} {r['embd']:>4} {r['weight_decay']:>6g} {r['steps']:>6} {r['batch']:>5} {r['params']:>7,} {r['train']:>6.3f} {r['val']:>6.3f} {r['gap']:>6.3f}")
    with open("results_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
