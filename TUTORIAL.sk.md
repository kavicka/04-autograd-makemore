# Tutoriál · 04 · P1: Autograd od nuly a makemore

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt v poradí, v akom vznikal: najprv malý autograd engine, potom testy, ktoré dokážu, že funguje správne, potom neurónová sieť nad ním a nakoniec znakový jazykový model v PyTorchi. Každý krok vysvetľuje **čo**, **prečo** a **ako**.

**Cieľ:** pochopiť, čo vlastne robí `loss.backward()`, tým, že si ho postavíš, a potom tie isté myšlienky použiť na tréning modelu, ktorý vymýšľa názvy slovenských obcí.

**Potrebuješ:** projekt 02 (reťazové pravidlo ručne) a projekt 03 (tvary).

---

## Časť A · Autograd engine

### Krok 1 · Číslo, ktoré si pamätá, odkiaľ prišlo

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                 # dL/dself, vyplní backward()
        self._backward = lambda: None   # posunie self.grad do detí
        self._prev = set(_children)     # Values, z ktorých toto vzniklo
        self._op = _op                  # "+", "*", ... na ladenie
```

**Prečo:** v projekte 02 sme backward pass napísali ručne pre jeden pevný vzorec. Aby bol automatický, každé číslo si musí pamätať, *z ktorých čísel vzniklo* (`_prev`) a *ako im poslať gradient späť* (`_backward`). Potom ľubovoľný vzorec, ktorý napíšeš, automaticky postaví graf, ktorým sa dá prejsť dozadu.

### Krok 2 · Každá operácia vypočíta hodnotu *aj* svoju lokálnu deriváciu

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

**Ako to funguje:** forward časť je len `self.data * other.data`. Closure `_backward` je reťazové pravidlo pre tento jeden uzol: „môj gradient (`out.grad`) krát moja lokálna derivácia ide každému rodičovi“. Je to closure, takže si pamätá `self`, `other` a `out`, kým sa backward nespustí.

**Prečo `+=` a nie `=`:** hodnotu môžeš použiť viackrát (`y = x*x + x`). Jej celkový gradient je *súčet* cez všetky cesty. S `=` by posledná cesta prepísala ostatné. Je to najčastejšia chyba v ručne písanom autograde a má vlastný test (`dy/dx = 2x + 1 = 7` pri x = 3).

**Prečo `other = ... Value(other)`:** aby si mohol písať `a * 2` alebo `a + 10` s obyčajnými číslami.

Engine implementuje len pár primitív: `+`, `*`, `**`, `exp`, `log`, `tanh`, `relu`, každú s deriváciou (`exp → e^x`, `log → 1/x`, `tanh → 1 − tanh²`, `relu → 1 ak kladné, inak 0`).

### Krok 3 · Zvyšok poskladaj z primitív

```python
def __neg__(self):          return self * -1
def __sub__(self, other):   return self + (-other)
def __truediv__(self, o):   return self * o**-1
def __radd__(self, other):  return self + other   # aby fungovalo 10 + a
```

**Prečo:** odčítanie, zápor a delenie sú kompozície `+`, `*` a `**`. Keď ich definuješ takto, dostanú správne gradienty zadarmo, bez ďalšieho kódu pre derivácie, v ktorom by sa dalo pomýliť. `__radd__`/`__rmul__` riešia prípad, keď je obyčajné číslo vľavo (`2 * a`), a umožnia, aby Pythonovské `sum()` fungovalo na `Value`.

### Krok 4 · `backward()`: prejdi graf v správnom poradí

```python
def backward(self):
    topo, visited = [], set()
    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)          # pridá sa až po všetkých svojich vstupoch
    build(self)
    self.grad = 1.0                 # dL/dL
    for v in reversed(topo):
        v._backward()
```

**Prečo topologické zoradenie:** uzol smie poslať gradient rodičom až vtedy, keď dostal gradient od *všetkého*, čo ho používa. Prehľadávanie do hĺbky `build` dá každý uzol za jeho vstupy; obrátený zoznam dá poradie, kde je každý uzol až po všetkých svojich používateľoch. Potom jeden prechod volaní `_backward()` je správny.

**Prečo `self.grad = 1.0`:** reťazové pravidlo potrebuje štart, `∂L/∂L = 1`.

Kontrola na príklade z projektu 02:
```python
a, b = Value(2.0), Value(-3.0)
L = (a * b + 10) ** 2
L.backward()     # a.grad = -24, b.grad = 16
```

### Krok 5 · Dokáž to testami proti PyTorchu ([`tests/test_engine.py`](tests/test_engine.py))

```bash
uv run pytest -v
```

Štyri testy, každý porovnáva s PyTorchom cez `torch.allclose` vo float64:
1. každá operácia (hodnota + oba gradienty),
2. zdieľaný uzol použitý dvakrát (chyba s `+=`),
3. dlhý výraz s 20+ operáciami,
4. celé 3-vrstvové MLP: gradient každej váhy oproti tej istej sieti postavenej v PyTorchi.

**Prečo sú tu testy dôležité:** zlá derivácia nespadne. Tréning je len o trochu horší a nikdy by si nezistil prečo. PyTorch je dôveryhodná referencia, takže zhoda s ním dokazuje správnosť.

---

## Časť B · Neurónová sieť nad enginom

### Krok 6 · Neuron → Layer → MLP ([`micrograd/nn.py`](micrograd/nn.py))

```python
class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
        return act.tanh() if self.nonlin else act
```

`Layer` je zoznam neurónov, `MLP` zoznam vrstiev. Každá vrstva má `tanh` okrem poslednej, ktorá vracia surové skóre.

**Prečo nelinearita:** bez `tanh` nemá zmysel skladať vrstvy: dve lineárne zobrazenia za sebou sú stále jedno lineárne zobrazenie, takže model by vedel nakresliť len priamku. `tanh` dovolí každej vrstve priestor ohnúť.

**Prečo je posledná vrstva lineárna:** výstup je skóre ľubovoľnej veľkosti; čo s ním, rozhodne loss.

**Prečo `Module.parameters()` a `zero_grad()`:** tréningový cyklus potrebuje zoznam všetkého, čo sa má aktualizovať, a medzi krokmi musí vynulovať gradienty (kvôli `+=` by sa inak sčítavali cez kroky).

### Krok 7 · Natrénuj ju na dvoch polmesiacoch ([`train_moons.py`](train_moons.py))

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

**Dáta:** 100 bodov na dvoch prepletených polmesiacoch, triedy +1/−1. Priamka ich neoddelí, takže to testuje, či sa sieť naozaj naučí zakrivenú hranicu.

**Prečo hinge loss:** `max(0, 1 − y·skóre)` je nula, keď je bod na správnej strane s rezervou aspoň 1. Tlačí len na body, ktoré sú zle alebo príliš blízko. Funguje s triedami ±1 a potrebuje len `relu`, ktoré engine má.

**Prečo L2 regularizácia:** pridá malú pokutu za veľké váhy, čo drží hranicu hladkú a nezabaľuje ju tesne okolo zašumených bodov.

**Prečo klesajúci learning rate (`1,0 → 0,1`):** veľké kroky na začiatku pre rýchly postup, malé na konci, aby sa model usadil a neskákal.

**Štvorriadkový cyklus** (forward → zero_grad → backward → update) je ten istý cyklus, aký používa tréning v PyTorchi v každom ďalšom projekte. Výsledok: 337 parametrov, 100 % presnosť na 100 trénovacích bodoch. Trvá to ~2 minúty, lebo každé číslo je Python objekt. Táto pomalosť je dôvod, prečo existujú tenzory.

---

## Časť C · makemore: znakový jazykový model ([`makemore.py`](makemore.py))

### Krok 8 · Priprav dáta

```python
words = [w for w in open("data/obce.txt", encoding="utf-8").read().splitlines() if w]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0          # jeden token značí začiatok aj koniec názvu
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
splits = {"train": words[:n1], "val": words[n1:n2], "test": words[n2:]}   # 80/10/10
```

- 2 750 názvov slovenských obcí, malými písmenami, 40 znakov + `.`.
- **`.` ako začiatočný aj koncový token:** model potrebuje vedieť, kde názov začína (aby vygeneroval prvé písmeno) a kedy skončiť. Jeden špeciálny token robí oboje.
- **Delenie podľa názvov, 80/10/10:** validačné názvy model pri tréningu nikdy nevidel, takže val loss meria skutočnú generalizáciu, nie pamäť. Test sa použije až na konci.

### Krok 9 · Vyber správny loss: negatívna log-vierohodnosť

Pre každý znak model vráti pravdepodobnosť pre každý možný nasledujúci znak. Loss je `−log P(správny ďalší znak)`, spriemerovaný.

**Prečo tento loss:** chceme, aby model dával vysokú pravdepodobnosť skutočným dátam (maximum likelihood). Pravdepodobnosť celého datasetu je súčin mnohých malých čísel, ktorý podtečie na 0. `log` premení súčin na súčet a mínus premení „maximalizuj“ na „minimalizuj“. `F.cross_entropy` v PyTorchi je presne toto.

**Baseline na porazenie:** model, ktorý nič nevie, háda rovnomerne zo 41 znakov: `log(41) = 3,714`. Každý užitočný model musí ísť pod túto hodnotu.

### Krok 10 · Baseline: bigramový model (čisté počítanie)

```python
N = torch.zeros((V, V))
for w in train_words:
    seq = ["."] + list(w) + ["."]
    for a, b in zip(seq, seq[1:]):
        N[stoi[a], stoi[b]] += 1
P = (N + 1) / (N + 1).sum(1, keepdim=True)
```

**Ako to funguje:** spočítaj, ako často každý znak nasleduje po každom inom, potom znormalizuj každý riadok na pravdepodobnosti. Generovanie = začni na `.`, vyžrebuj ďalší znak z toho riadku, opakuj až po `.`.

**Prečo vyhladenie +1:** dvojica, ktorá v tréningu nebola, by dostala pravdepodobnosť 0 a `log 0 = −∞` na validačných dátach. Pripočítanie 1 ku každému počtu dá nevideným dvojiciam malú pravdepodobnosť.

**Prečo vôbec baseline:** bez neho nevieš povedať, či je neurónová sieť dobrá. Bigram má val NLL 2,462. Jeho vzorky majú správne písmená, ale žiadnu štruktúru (*hysučíké sovopováňé*), lebo jeden znak kontextu je primálo.

### Krok 11 · Postav dataset pre MLP posuvným oknom

```python
def build_dataset(ws):
    for w in ws:
        context = [0] * BLOCK            # štart: "..."
        for ch in w + ".":
            X.append(context); Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
```

Pre „trnava“ s `BLOCK = 3`: `... → t`, `..t → r`, `.tr → n`, `trn → a`, … Z každého názvu vznikne veľa trénovacích príkladov (3 znaky → ďalší znak).

### Krok 12 · Model: embeddingy + jedna skrytá vrstva (Bengio a kol., 2003)

```python
emb    = C[X]                                         # (B, 3)  → (B, 3, 10)
h      = tanh(emb.view(-1, 30) @ W1 + b1)             # (B, 30) → (B, 200)
logits = h @ W2 + b2                                  # (B, 200) → (B, 41)
```

- **Tabuľka embeddingov `C` (41 × 10):** každý znak dostane naučiteľný vektor 10 čísel. Podobné znaky môžu skončiť s podobnými vektormi, takže čo sa model naučí o jednom, prenesie sa aj na ostatné.
- **`view(-1, 30)`:** zlep tri 10-rozmerné vektory do jedného 30-rozmerného vstupu.
- **Skrytá vrstva 200 tanh jednotiek**, potom **logity** (jedno skóre pre každý možný ďalší znak). Tvary presne podľa pravidiel z projektu 03.

### Krok 13 · Inicializuj opatrne

```python
W1 = torch.randn((30, 200)) * (5/3) / 30**0.5    # Kaiming init pre tanh
W2 = torch.randn((200, V)) * 0.01                # malinké → takmer rovnomerný štart
```

**Prečo malinké `W2`:** na začiatku má byť model *neistý* a predpovedať takmer rovnomerne. Potom je počiatočný loss ≈ `log(41) = 3,71`. S veľkými náhodnými váhami by sa model sebaisto mýlil, prvý loss by bol oveľa vyšší a prvých tisíc krokov by sa minulo na to, aby sa to napravilo.

**Prečo Kaiming škálovanie pre `W1`:** keď sú pre-aktivácie príliš veľké, `tanh` sa nasýti na ±1, jeho derivácia `1 − tanh²` je ≈ 0 a gradienty prestanú tiecť. Delenie √fan_in ich udrží v užitočnom rozsahu tanh; 5/3 je zosilnenie, ktoré kompenzuje stláčanie tanh.

**Kontrola:** prvý zaznamenaný loss je 3,70, presne pri `log(41)`.

### Krok 14 · Zvyk pri ladení: najprv preuč jeden batch

```python
Xb, Yb = data["train"][0][:32], data["train"][1][:32]
for _ in range(1000):
    loss = F.cross_entropy(forward(params, Xb), Yb)
    ...
```

**Prečo:** správny model so 14 851 parametrami si musí vedieť zapamätať 32 príkladov. Loss išiel 3,72 → 0,26. Keď nejde blízko nuly, je chyba vo forward, backward alebo update a žiadne ladenie hyperparametrov to neopraví. Keď to spravíš ako prvé, ušetríš hodiny.

### Krok 15 · Trénuj s minibatchmi, weight decay a klesajúcim LR

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

- **Minibatche po 32:** gradient z 32 náhodných príkladov je zašumený, ale lacný odhad celého gradientu. Veľa lacných krokov porazí pár presných.
- **Skokový pokles LR po 2/3:** to je viditeľný schod na krivke lossu.
- **Weight decay 1e-3** na `W1`, `W2` vybral sweep (ďalší krok).
- `p.grad = None` je `zero_grad` v PyTorchi.

### Krok 16 · Hyperparametre vyber podľa validačného lossu ([`sweep.py`](sweep.py))

Sweep natrénuje šesť konfigurácií. Kľúčová dvojica riadkov:

| konfigurácia | train | val |
|---|---:|---:|
| 200 skrytých, bez weight decay | 1,432 | 2,002 |
| 200 skrytých, weight decay 1e-3 | 1,704 | **1,952** |

**Čo to ukazuje:** s len 2 200 trénovacími názvami sa väčší model **preučí**: train loss stále klesá, kým val stojí. Weight decay spraví train *horší*, ale val *lepší*. Pravidlo: vyberaj vždy podľa validácie, nikdy podľa trénovacieho lossu.

### Krok 17 · Generuj názvy s teplotou

```python
probs = F.softmax(logits / temperature, dim=1)
ix = torch.multinomial(probs, 1).item()
```

**Ako funguje teplota:** delenie logitov T < 1 spraví veľké logity relatívne ešte väčšími, takže softmax je ostrejší: typickejšie, menej kreatívne názvy. T = 1 vzorkuje z naučeného rozdelenia tak, ako je.

- T = 1,0: *brašná kamená cirovce, dolný tirtoň, malý sačovce*
- T = 0,7: *plavská nad štiarom, malá ves, klenica, hladice*

Skript tiež overí, že 99 % z 500 vzoriek sú **nové** názvy, takže model generalizuje, nekopíruje.

### Krok 18 · Pozri sa dnu: embeddingy

`plot.py` premietne 10-rozmerné embeddingy znakov do 2-D. Samohlásky *a, i, e* sú spolu, *u* a *o* sedia vedľa seba a veľa spoluhlások je inde. Nikto modelu nepovedal, čo je samohláska; zoskupenie objavil sám, lebo pomáha predpovedať ďalší znak.

---

## Vyskúšaj sám

1. Pridaj do `Value` `sigmoid` s deriváciou `s·(1 − s)` a test proti `torch.sigmoid`.
2. Zmeň `BLOCK` na 5. Zlepší sa val NLL, alebo sa pri takto málo dátach model viac preučí?
3. Odstráň `* 0.01` z `W2` a vypíš prvý loss. O koľko je nad 3,71?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1–4 | Autograd = graf hodnôt + lokálna derivácia pre každú operáciu + spätný topologický prechod |
| 2 | Gradienty sa sčítavajú cez `+=` |
| 5 | Testuj gradienty proti dôveryhodnej referencii |
| 6–7 | Nelinearita + štvorriadkový tréningový cyklus |
| 9–10 | NLL = maximum likelihood; porazi `log(V)` a počítací baseline |
| 13–14 | Skontroluj počiatočný loss, preuč jeden batch pred skutočným tréningom |
| 16 | Vyberaj podľa validačného lossu; weight decay bojuje proti preučeniu |
| 17 | Teplota vymieňa kreativitu za typickosť |
