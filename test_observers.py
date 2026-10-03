"""CPU test of the observer design (no GPU, no data needed).

Gate (a) in miniature: a toy training loop with shuffled DataLoader + dropout.
Run it with observers OFF and ON; trainer RNG fingerprints and losses must be identical.
Negative control: an observer WITHOUT isolated_rng must change them (proves the test can fail).
"""
import os, random
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from repro_utils import isolated_rng, observer, rng_fp, PLACEMENT_SEED


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def insert_trigger(text, trigger, rng=random):
    words = text.split()
    pos = rng.randint(1, len(words) - 1)
    return " ".join(words[:pos] + [trigger] + words[pos:])


def observe(model, loader, texts, mode):
    """Stand-in for test acc + ASR: iterates a shuffled loader, uses dropout, places triggers."""
    def body():
        model.eval() if mode != "dropout_on" else model.train()
        for x, _ in loader:
            model(x)
        rng = random.Random(PLACEMENT_SEED)
        return [insert_trigger(t, "mn", rng) for t in texts]
    if mode == "isolated":
        with observer(model):
            return body()
    if mode == "rng_only":      # isolates RNG but leaves model in eval(): must be caught
        with isolated_rng():
            return body()
    return body()


def run(observer_mode, epochs=4):
    set_all_seeds(0)
    X, Y = torch.randn(256, 8), torch.randint(0, 2, (256,))
    ds = TensorDataset(X, Y)
    train = DataLoader(ds, batch_size=32, shuffle=True)
    obs = DataLoader(ds, batch_size=64, shuffle=True)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(0.1), torch.nn.Linear(16, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    texts = ["a b c d e f g h"] * 5
    fps, losses, placed = [], [], None
    model.train()   # trainer sets train mode ONCE (worst case: it never re-asserts it per epoch)
    for ep in range(epochs):
        fps.append(rng_fp())
        tot = 0.0
        for x, y in train:
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(model(x), y)
            l.backward(); opt.step(); tot += l.item()
        losses.append(round(tot, 6))
        if observer_mode != "off":
            placed = observe(model, obs, texts, observer_mode)
    fps.append(rng_fp())
    return fps, losses, placed


off = run("off")
iso = run("isolated")
raw = run("raw")
rngonly = run("rng_only")
print("off      :", off[0], off[1])
print("isolated :", iso[0], iso[1])
print("raw      :", raw[0], raw[1])
print("rng_only :", rngonly[0], rngonly[1])
assert off[0] == iso[0] and off[1] == iso[1], "FAIL: isolated observer moved trainer RNG"
assert off[1] != rngonly[1], "FAIL: eval-mode leak not detected (test vacuous)"
assert off[0] != raw[0], "FAIL: negative control did not perturb (test is vacuous)"
# placement is a pure function of PLACEMENT_SEED, identical every epoch/run
a = [insert_trigger("a b c d e f g h", "mn", random.Random(PLACEMENT_SEED)) for _ in range(2)]
assert a[0] == a[1]
print("placement:", a[0])
print("PASS: observer neutrality holds; negative control perturbs as expected")