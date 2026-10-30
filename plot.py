"""Draws every figure in assets/ from the results_*.json files."""

import json

import matplotlib.pyplot as plt
import numpy as np

M = json.load(open("results_moons.json"))
K = json.load(open("results_makemore.json", encoding="utf-8"))
BLUE, ORANGE, GRAY, RED = "#2563eb", "#ea580c", "#6b7280", "#dc2626"


def moons() -> None:
    X, y = np.array(M["X"]), np.array(M["y"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    xx, yy = np.meshgrid(M["grid_x"], M["grid_y"])
    a1.contourf(xx, yy, np.array(M["grid"]) > 0, levels=1, colors=["#fed7aa", "#bfdbfe"], alpha=0.7)
    a1.contour(xx, yy, np.array(M["grid"]), levels=[0], colors="black", linewidths=1)
    a1.scatter(X[y > 0, 0], X[y > 0, 1], c=BLUE, s=25, edgecolors="white", label="class +1")
    a1.scatter(X[y < 0, 0], X[y < 0, 1], c=ORANGE, s=25, edgecolors="white", label="class −1")
    a1.set(title=f"Decision boundary of our MLP ({M['n_params']} params, pure-Python autograd)", xticks=[], yticks=[])
    a1.legend(loc="lower left", fontsize=9)

    h = M["history"]
    steps = [r["step"] for r in h]
    a2.plot(steps, [r["loss"] for r in h], color=BLUE, label="loss (hinge + L2)")
    a2.set(xlabel="step", ylabel="loss", title="Training with our own backward()")
    a2b = a2.twinx()
    a2b.plot(steps, [r["acc"] * 100 for r in h], color=ORANGE, label="accuracy")
    a2b.set(ylabel="accuracy [%]", ylim=(0, 105))
    a2.legend(loc="center right", fontsize=9)
    a2b.legend(loc="right", bbox_to_anchor=(1, 0.38), fontsize=9)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/moons.png", dpi=150)


def loss_curves() -> None:
    c = K["curve"]
    steps = [r["step"] for r in c]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(steps, [r["train"] for r in c], color=BLUE, label=f"MLP train (final {K['mlp']['train']:.3f})")
    ax.plot(steps, [r["val"] for r in c], color=ORANGE, label=f"MLP val (final {K['mlp']['val']:.3f})")
    ax.axhline(K["bigram"]["val"], color=GRAY, ls="--", label=f"bigram val ({K['bigram']['val']:.3f})")
    ax.axhline(K["uniform"], color=RED, ls=":", label=f"uniform guess ({K['uniform']:.3f})")
    ax.axvline(20_000, color="black", alpha=0.2)
    ax.text(20_300, 3.4, "lr 0.1 → 0.01", fontsize=8, color=GRAY)
    ax.set(xlabel="step", ylabel="NLL per character", title="Makemore on Slovak municipality names", ylim=(1.4, 3.8))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/makemore_loss.png", dpi=150)


def bigram_heatmap() -> None:
    N = np.array(K["bigram_counts"])
    chars = K["chars"]
    fig, ax = plt.subplots(figsize=(10, 9.4))
    im = ax.imshow(np.log1p(N), cmap="Blues")
    ax.set_xticks(range(len(chars)), [c if c != " " else "␣" for c in chars], fontsize=8)
    ax.set_yticks(range(len(chars)), [c if c != " " else "␣" for c in chars], fontsize=8)
    ax.set(xlabel="next character", ylabel="current character", title="Bigram counts (log scale): which character follows which")
    fig.colorbar(im, ax=ax, shrink=0.75, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig("assets/bigram_heatmap.png", dpi=150)


def embeddings() -> None:
    E = np.array(K["embeddings"])
    E = E - E.mean(0)
    _, _, Vt = np.linalg.svd(E, full_matrices=False)
    P = E @ Vt[:2].T  # project 10-D embeddings onto their 2 main directions (PCA)
    vowels = set("aáäeéiíoóôuúyý")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for (x, yv), ch in zip(P, K["chars"]):
        col = RED if ch in vowels else (GRAY if ch in ". -" else BLUE)
        ax.scatter(x, yv, s=260, color=col, alpha=0.15)
        ax.text(x, yv, ch if ch != " " else "␣", ha="center", va="center", fontsize=11, color=col)
    ax.set(title="Learned character embeddings (PCA of 10-D → 2-D)\nred = vowels, blue = consonants, gray = '.', space, '-'", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig("assets/embeddings.png", dpi=150)


def overfit() -> None:
    o = K["overfit_one_batch"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(o, color=BLUE)
    ax.set(xlabel="step", ylabel="loss", title=f"Sanity check: overfit one batch of 32 ({o[0]:.2f} → {o[-1]:.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/overfit_one_batch.png", dpi=150)


if __name__ == "__main__":
    moons()
    loss_curves()
    bigram_heatmap()
    embeddings()
    overfit()
