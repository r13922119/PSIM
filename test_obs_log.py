"""CPU test of obs_log.Observer (no GPU, no data). Run:  PYTHONNOUSERSITE=1 python test_obs_log.py

Checks, on a toy model that has the same module layout as the peft LoRA model:
  1. every logged number equals the same number computed from a dense dW = s * B @ A;
  2. pen_k equals the training-loop penalty formula in variant_finetune.py;
  3. U_k, V_k really are the top singular vectors of W_pre;
  4. a snapshot changes no RNG stream, no parameter, no gradient;
  5. files have the expected number of rows, and the B = 0 start (epoch 0, step 0) gives NaN shares, not a crash.
"""
import csv, math, os, shutil, sys, tempfile
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
import torch.nn as nn
from repro_utils import rng_fp
from obs_log import Observer

D, R, KS, SCALE, NL = 64, 8, [4, 16], 2.0, 4
torch.manual_seed(0)


class LoraLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(D, D, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(D, R, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(R, D, bias=False)})
        self.scaling = {"default": SCALE}
        nn.init.zeros_(self.lora_B["default"].weight)     # peft starts with B = 0

    def forward(self, x):
        return self.base_layer(x) + SCALE * self.lora_B["default"](self.lora_A["default"](x))


class SelfAtt(nn.Module):
    def __init__(self):
        super().__init__(); self.query = LoraLinear(); self.value = LoraLinear()
    def forward(self, x): return self.query(x) + self.value(x)

class Att(nn.Module):
    def __init__(self):
        super().__init__(); self.self = SelfAtt()
    def forward(self, x): return self.self(x)

class Layer(nn.Module):
    def __init__(self):
        super().__init__(); self.attention = Att()
    def forward(self, x): return torch.tanh(self.attention(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = nn.Module(); self.encoder.layer = nn.ModuleList([Layer() for _ in range(NL)])
    def forward(self, x):
        for l in self.encoder.layer: x = l(x)
        return x


def main():
    model = Model()
    out = tempfile.mkdtemp()
    obs = Observer(model, ["query", "value"], out, KS, every=2)
    assert len(obs.mods) == 2 * NL
    # 3. U_k, V_k are singular vectors of W_pre
    for i, (_, m) in enumerate(obs.mods):
        W = m.base_layer.weight.detach().double()
        S = torch.linalg.svdvals(W)
        UtWV = obs.U[i][:, :max(KS)].T @ W @ obs.V[i][:, :max(KS)]
        assert torch.allclose(torch.diagonal(UtWV), S[:max(KS)], rtol=1e-4), "U/V are not the top singular vectors"
        assert float(UtWV.abs().sum() - torch.diagonal(UtWV).abs().sum()) < 1e-3, "off-diagonal mass: U/V mismatch"
    print("ok 3: U_k, V_k are the top singular vectors of W_pre")

    opt = torch.optim.SGD([p for n, p in model.named_parameters() if "lora_" in n], lr=0.5)
    x = torch.randn(16, D)
    n_snap = 0
    for step in range(6):
        opt.zero_grad()
        loss = (model(x) ** 2).mean()
        obs.accumulate_task(loss.detach())
        obs.accumulate_pen(torch.tensor(0.123))
        loss.backward()
        if obs.due(step, 5):
            before_rng = rng_fp()
            before_p = [p.detach().clone() for p in model.parameters()]
            before_g = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
            obs.snapshot(0, step)
            n_snap += 1
            assert rng_fp() == before_rng, "snapshot moved an RNG stream"
            assert all(torch.equal(a, b) for a, b in zip(before_p, model.parameters())), "snapshot changed a parameter"
            assert all((g is None and p.grad is None) or torch.equal(g, p.grad) for g, p in zip(before_g, model.parameters())), "snapshot changed a gradient"
        opt.step()
    obs.epoch_end(0, 0.9, 0.91, 0.5)
    print("ok 4: a snapshot moves no RNG stream and changes no parameter or gradient")
    obs.write_summary(dict(best_epoch=3, pt_scales={"x": dict(new_scaling=1.5)}, args={"seed": 3}))
    import json as _j; assert _j.load(open(os.path.join(out, "summary.json")))["pt_scales"]["x"]["new_scaling"] == 1.5
    print("ok: summary.json written and readable")

    # 1 + 2: dense recomputation from the final state (the epoch_end snapshot, step == 'end')
    rows = list(csv.DictReader(open(os.path.join(out, "modules.csv"))))
    end = [r for r in rows if r["step"] == "end"]
    assert len(end) == 2 * NL
    pen_k = {k: [] for k in KS}
    pen_k3 = {k: [] for k in KS}      # same, last 3 layers only
    for i, (name, m) in enumerate(obs.mods):
        r = next(x for x in end if (int(x["layer"]), x["mod"]) == (int(name.split("layer.")[1].split(".")[0]), name.split(".")[-1]))
        A, B = m.lora_A["default"].weight.detach().double(), m.lora_B["default"].weight.detach().double()
        dW = SCALE * B @ A                                   # dense reference, d x d
        sig_pre = float(torch.linalg.svdvals(m.base_layer.weight.detach().double())[0])
        want = {"fro": float(dW.norm()), "rel": float(torch.linalg.svdvals(dW)[0]) / sig_pre,
                "normA2": float((A * A).sum()), "normB2": float((B * B).sum())}
        for k in KS:
            U, V = obs.U[i][:, :k], obs.V[i][:, :k]
            want[f"R{k}"] = float(((dW @ V) ** 2).sum() / (dW ** 2).sum())
            want[f"L{k}"] = float(((U.T @ dW) ** 2).sum() / (dW ** 2).sum())
            want[f"wA{k}"] = (D / k) * float(((A @ V) ** 2).sum() / (A ** 2).sum())
            want[f"wB{k}"] = (D / k) * float(((U.T @ B) ** 2).sum() / (B ** 2).sum())
            # the penalty exactly as the training loop writes it (float32 tensors)
            Uf, Vf, Bf, Af = U.float(), V.float(), B.float(), A.float()
            omega_B = (Uf.T @ Bf).pow(2).sum() * (Uf.shape[0] / Uf.shape[1])
            omega_A = (Af @ Vf).pow(2).sum() * (Vf.shape[0] / Vf.shape[1])
            pen_k[k].append(float(omega_A + omega_B))
            if int(name.split("layer.")[1].split(".")[0]) >= NL - 3:
                pen_k3[k].append(float(omega_A + omega_B))
        for key, v in want.items():
            got = float(r[key])
            assert math.isclose(got, v, rel_tol=1e-5, abs_tol=1e-9), f"{name} {key}: logged {got} vs dense {v}"
    print("ok 1: every per-module number equals the dense computation")
    ep = list(csv.DictReader(open(os.path.join(out, "epochs.csv"))))[0]
    for k in KS:
        want = sum(pen_k[k]) / len(pen_k[k])
        assert math.isclose(float(ep[f"pen{k}"]), want, rel_tol=1e-4), f"pen{k}: {ep[f'pen{k}']} vs training formula {want}"
        want3 = sum(pen_k3[k]) / len(pen_k3[k])
        assert len(pen_k3[k]) == 6 and math.isclose(float(ep[f"pen{k}_top3"]), want3, rel_tol=1e-4), f"pen{k}_top3: {ep[f'pen{k}_top3']} vs {want3}"
    assert math.isclose(float(ep["train_penalty"]), 0.123, rel_tol=1e-6) and ep["dev"] and float(ep["asr"]) == 0.5
    print("ok 2: pen_k equals the training-loop penalty formula (mean over modules)")

    # 5. row counts and the B = 0 start
    steps = list(csv.DictReader(open(os.path.join(out, "steps.csv"))))
    assert len(steps) == n_snap == 4, (len(steps), n_snap)             # steps 0, 2, 4, and last (5)
    assert len(rows) == (n_snap + 1) * 2 * NL
    first = [r for r in rows if r["step"] == "0"]
    assert all(float(r["rel"]) == 0.0 and r["R4"] == "nan" and r["wB4"] == "nan" for r in first), "B=0 start should give rel=0 and NaN shares"
    assert not math.isnan(float(first[0]["wA4"])), "A is nonzero at the start, wA must exist"
    print("ok 5: row counts correct; B = 0 start gives rel = 0 and NaN shares without crashing")
    shutil.rmtree(out)
    print("ALL OK")


main()
