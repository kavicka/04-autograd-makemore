# Tutorial · 04 · P1: Autograd from scratch & makemore

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial rebuilds the project in the order it was written: first a tiny autograd engine, then tests that prove it is right, then a neural network on top of it, and finally a character-level language model in PyTorch. Every step explains **what**, **why** and **how**.

**Goal:** understand what `loss.backward()` actually does by building it, then use the same ideas to train a model that invents Slovak municipality names.

**You need:** project 02 (chain rule by hand) and project 03 (shapes).

---

## Part A · The autograd engine

### Step 1 · A number that remembers where it came from

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                 # dL/dself, filled in by backward()
        self._backward = lambda: None   # pushes self.grad into the children
        self._prev = set(_children)     # the Values this one was made from
        self._op = _op                  # "+", "*", ... for debugging
```

**Why:** in project 02 we wrote the backward pass by hand for one fixed formula. To make it automatic, every number must remember *which numbers produced it* (`_prev`) and *how to pass a gradient back to them* (`_backward`). Then any formula you write automatically builds a graph you can walk backwards.

### Step 2 · Each operation computes a value *and* its local derivative

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    def _backward():
        self.grad  += other.data * out.grad   # d(a*b)/da = b
        other.grad += self.data  * out.grad   # d(a*b)/db = a
    out._backward = _backward
    return out
```

**How it works:** the forward part is just `self.data * other.data`. The `_backward` closure is the chain rule for this one node: "my gradient (`out.grad`) times my local derivative goes to each parent". It is a closure, so it remembers `self`, `other` and `out` until backward runs.

**Why `+=` and not `=`:** a value can be used more than once (`y = x*x + x`). Its total gradient is the *sum* over all paths. With `=`, the last path would overwrite the others. This is the most common bug in hand-written autograd, and there is a dedicated test for it (`dy/dx = 2x + 1 = 7` at x = 3).

**Why `other = ... Value(other)`:** so you can write `a * 2` or `a + 10` with plain numbers.

The engine implements only a few primitives: `+`, `*`, `**`, `exp`, `log`, `tanh`, `relu`, each with its derivative (`exp → e^x`, `log → 1/x`, `tanh → 1 − tanh²`, `relu → 1 if positive else 0`).

### Step 3 · Build the rest from the primitives

```python
def __neg__(self):          return self * -1
def __sub__(self, other):   return self + (-other)
def __truediv__(self, o):   return self * o**-1
def __radd__(self, other):  return self + other   # makes 10 + a work
```

**Why:** subtraction, negation and division are compositions of `+`, `*` and `**`. Defining them this way means they get correct gradients for free, with no extra derivative code to get wrong. `__radd__`/`__rmul__` handle the case where the plain number is on the left (`2 * a`), and they let Python's `sum()` work on `Value`s.

### Step 4 · `backward()`: walk the graph in the right order

```python
def backward(self):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)          # appended only after all its inputs
    build(self)
    self.grad = 1.0                 # dL/dL
    for v in reversed(topo):
        v._backward()
```

**Why a topological sort:** a node may only push its gradient to its parents once it has received gradient from *everything* that uses it. The depth-first `build` puts every node after its inputs; reversing that list gives an order where each node comes after all its users. Then a single pass of `_backward()` calls is correct.

**Why `self.grad = 1.0`:** the chain rule needs a starting point, `∂L/∂L = 1`.

Check with project 02's example:
```python
a, b = Value(2.0), Value(-3.0)
L = (a * b + 10) ** 2
L.backward()     # a.grad = -24, b.grad = 16
```

### Step 5 · Prove it with tests against PyTorch ([`tests/test_engine.py`](tests/test_engine.py))

```bash
uv run pytest -v
```

Four tests, each comparing to PyTorch with `torch.allclose` in float64:
1. every operation (value + both gradients),
2. a shared node used twice (the `+=` bug),
3. a long expression with 20+ operations,
4. a whole 3-layer MLP: every weight's gradient vs the same network rebuilt in PyTorch.

**Why tests matter here:** a wrong derivative does not crash. Training just gets slightly worse, and you would never know why. PyTorch is a trusted reference, so matching it proves correctness.

---

## Part B · A neural network on the engine

### Step 6 · Neuron → Layer → MLP ([`micrograd/nn.py`](micrograd/nn.py))

```python
class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
        return act.tanh() if self.nonlin else act
```

`Layer` is a list of neurons, `MLP` a list of layers. Every layer has `tanh` except the last, which outputs a raw score.

**Why the non-linearity:** without `tanh`, stacking layers is pointless: two linear maps in a row are still one linear map, so the model could only draw a straight line. `tanh` lets each layer bend the space.

**Why the last layer is linear:** the output is a score that can be any size; the loss decides what to do with it.

**Why `Module.parameters()` and `zero_grad()`:** the training loop needs a list of everything to update, and must reset gradients between steps (because of `+=`, they would otherwise accumulate across steps).

### Step 7 · Train it on two half-moons ([`train_moons.py`](train_moons.py))

```python
scores = [model([Value(a), Value(b)]) for a, b in X]
data_loss = sum((1 + -yi * s).relu() for yi, s in zip(y, scores)) / len(y)   # hinge
reg_loss  = alpha * sum(p * p for p in model.parameters())                  # L2
loss = data_loss + reg_loss

model.zero_grad()
loss.backward()
for p in model.parameters():
    p.data -= lr * p.grad
```

**The data:** 100 points on two interleaved half-moons, labels +1/−1. A straight line cannot separate them, so this tests whether the network really learns a curved boundary.

**Why hinge loss:** `max(0, 1 − y·score)` is zero once a point is on the correct side with a margin of at least 1. It only pushes on points that are wrong or too close. It works with labels ±1 and needs only `relu`, which the engine has.

**Why L2 regularisation:** it adds a small penalty on large weights, which keeps the boundary smooth instead of wrapping tightly around noisy points.

**Why a learning-rate decay (`1.0 → 0.1`):** big steps early to make progress, small steps late to settle without bouncing.

**The four-line loop** (forward → zero_grad → backward → update) is the same loop PyTorch training uses in every later project. Result: 337 parameters, 100 % accuracy on the 100 training points. It takes ~2 minutes, because every number is a Python object. That slowness is the motivation for tensors.

---

## Part C · makemore: a character-level language model ([`makemore.py`](makemore.py))

### Step 8 · Prepare the data

```python
words = [w for w in open("data/obce.txt", encoding="utf-8").read().splitlines() if w]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0          # one token marks both start and end of a name
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
splits = {"train": words[:n1], "val": words[n1:n2], "test": words[n2:]}   # 80/10/10
```

- 2,750 Slovak municipality names, lowercased, 40 characters + `.`.
- **`.` as start and end token:** the model needs to know where a name begins (to generate the first letter) and when to stop. One special token does both.
- **Split by name, 80/10/10:** validation names were never seen in training, so the val loss measures real generalisation, not memory. Test is touched only at the end.

### Step 9 · Pick the right loss: negative log-likelihood

For each character, the model outputs a probability for every possible next character. The loss is `−log P(correct next char)`, averaged.

**Why this loss:** we want the model to assign high probability to the real data (maximum likelihood). The probability of a whole dataset is a product of many small numbers, which underflows to 0. Taking `log` turns the product into a sum, and the minus sign turns "maximise" into "minimise". PyTorch's `F.cross_entropy` is exactly this.

**Baseline to beat:** a model that knows nothing guesses uniformly over 41 characters: `log(41) = 3.714`. Any useful model must go below that.

### Step 10 · Baseline: the bigram model (pure counting)

```python
N = torch.zeros((V, V))
for w in train_words:
    seq = ["."] + list(w) + ["."]
    for a, b in zip(seq, seq[1:]):
        N[stoi[a], stoi[b]] += 1
P = (N + 1) / (N + 1).sum(1, keepdim=True)
```

**How it works:** count how often each character follows each other one, then normalise every row into probabilities. Generating = start at `.`, sample the next char from that row, repeat until `.`.

**Why +1 smoothing:** a pair never seen in training would get probability 0, and `log 0 = −∞` on the validation set. Adding 1 to every count gives unseen pairs a small probability.

**Why a baseline at all:** without one you cannot say whether the neural network is good. Bigram scores val NLL 2.462. Its samples have the right letters but no structure (*hysučíké sovopováňé*) because one character of context is too little.

### Step 11 · Build the MLP dataset with a sliding window

```python
def build_dataset(ws):
    for w in ws:
        context = [0] * BLOCK            # start: "..."
        for ch in w + ".":
            X.append(context); Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
```

For "trnava" with `BLOCK = 3`: `... → t`, `..t → r`, `.tr → n`, `trn → a`, … Every name becomes many (3 chars → next char) training examples.

### Step 12 · The model: embeddings + one hidden layer (Bengio et al., 2003)

```python
emb    = C[X]                                         # (B, 3)  → (B, 3, 10)
h      = tanh(emb.view(-1, 30) @ W1 + b1)             # (B, 30) → (B, 200)
logits = h @ W2 + b2                                  # (B, 200) → (B, 41)
```

- **Embedding table `C` (41 × 10):** each character gets a learnable 10-number vector. Similar characters can end up with similar vectors, so what the model learns about one transfers to others.
- **`view(-1, 30)`:** glue the three 10-D vectors into one 30-D input.
- **Hidden layer of 200 tanh units**, then **logits** (one score per possible next character). Shapes follow project 03's rules exactly.

### Step 13 · Initialise carefully

```python
W1 = torch.randn((30, 200)) * (5/3) / 30**0.5    # Kaiming init for tanh
W2 = torch.randn((200, V)) * 0.01                # tiny → near-uniform start
```

**Why `W2` tiny:** at the start the model should be *unsure*, predicting nearly uniform probabilities. Then the initial loss is ≈ `log(41) = 3.71`. With large random weights the model would be confidently wrong, the first loss would be much higher, and the first thousand steps would be wasted undoing that.

**Why Kaiming scaling for `W1`:** if the pre-activations are too large, `tanh` saturates at ±1, its derivative `1 − tanh²` becomes ≈ 0, and gradients stop flowing. Dividing by √fan_in keeps them in tanh's useful range; 5/3 is the gain that compensates for tanh squashing.

**Check:** the first logged loss is 3.70, right at `log(41)`.

### Step 14 · Debug habit: overfit one batch first

```python
Xb, Yb = data["train"][0][:32], data["train"][1][:32]
for _ in range(1000):
    loss = F.cross_entropy(forward(params, Xb), Yb)
    ...
```

**Why:** a correct model with 14,851 parameters must be able to memorise 32 examples. Loss went 3.72 → 0.26. If it does not go near zero, there is a bug in forward, backward or update, and no hyperparameter tuning will fix it. Doing this first saves hours.

### Step 15 · Train with minibatches, weight decay and LR decay

```python
for i in range(30_000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    loss = F.cross_entropy(forward(params, Xtr[ix]), Ytr[ix])
    reg  = 1e-3 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    for p in params: p.grad = None
    (loss + reg).backward()
    lr = 0.1 if i < 20_000 else 0.01
    for p in params: p.data += -lr * p.grad
```

- **Minibatches of 32:** the gradient from 32 random examples is a noisy but cheap estimate of the full gradient. Many cheap steps beat a few exact ones.
- **Step LR decay at 2/3:** that is the visible drop in the loss curve.
- **Weight decay 1e-3** on `W1`, `W2` was chosen by the sweep (next step).
- `p.grad = None` is PyTorch's `zero_grad`.

### Step 16 · Choose hyperparameters by validation loss ([`sweep.py`](sweep.py))

The sweep trains six configurations. The key row pair:

| config | train | val |
|---|---:|---:|
| 200 hidden, no weight decay | 1.432 | 2.002 |
| 200 hidden, weight decay 1e-3 | 1.704 | **1.952** |

**What it shows:** with only 2,200 training names, the bigger model **overfits**: train loss keeps dropping while val stalls. Weight decay makes train *worse* but val *better*. The rule: always select by validation, never by training loss.

### Step 17 · Sample names with a temperature

```python
probs = F.softmax(logits / temperature, dim=1)
ix = torch.multinomial(probs, 1).item()
```

**How temperature works:** dividing logits by T < 1 makes large logits relatively larger, so softmax becomes sharper: more typical, less creative names. T = 1 samples from the learned distribution as-is.

- T = 1.0: *brašná kamená cirovce, dolný tirtoň, malý sačovce*
- T = 0.7: *plavská nad štiarom, malá ves, klenica, hladice*

The script also checks that 99 % of 500 samples are **new** names, so the model generalises rather than copies.

### Step 18 · Look inside: the embeddings

`plot.py` projects the 10-D character embeddings to 2-D. Vowels *a, i, e* cluster together, *u* and *o* sit together, and many consonants gather elsewhere. Nobody told the model what a vowel is; it discovered that grouping because it helps predict the next character.

---

## Try it yourself

1. Add `sigmoid` to `Value` with its derivative `s·(1 − s)` and a test against `torch.sigmoid`.
2. Change `BLOCK` to 5. Does val NLL improve, or does it overfit more with this little data?
3. Remove the `* 0.01` from `W2` and print the first loss. How far above 3.71 is it?

## Recap

| Step | Idea |
|---|---|
| 1–4 | Autograd = graph of values + local derivative per op + reverse topological walk |
| 2 | Gradients accumulate with `+=` |
| 5 | Test gradients against a trusted reference |
| 6–7 | Non-linearity + the four-line training loop |
| 9–10 | NLL = maximum likelihood; beat `log(V)` and a counting baseline |
| 13–14 | Check the initial loss, overfit one batch before real training |
| 16 | Select by validation loss; weight decay fights overfitting |
| 17 | Temperature trades creativity for typicality |
