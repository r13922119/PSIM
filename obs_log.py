"""obs_log.py: read-only, per-module observer of the LoRA update during training (Tr diagnostics).

dw_tr.py measures the LoRA update once, offline, at the best-dev checkpoint. This logs the same quantities
DURING training: every --obs_every steps and at the end of every epoch, for every q/v module, in every arm
(baseline / Cl / Tr), with one definition shared by all arms (so a baseline run shows what the Tr penalty
WOULD have been, and a Tr run's pen_k at its own k can be checked against the [DEBUG] lines in the log).

Quantities per module (A: r x d_in, B: d_out x r, s = module.scaling, dW = s * B @ A, never formed):
  rel      sigma_max(dW) / sigma_max(W_pre)      (1 / rel is the factor Pt applies)
  fro      ||dW||_F
  R_k, L_k share of ||dW||_F^2 inside the top-k RIGHT (input side) / LEFT (output side) singular directions of W_pre;
           a uniformly random update would give k/d
  wA_k     (d_in/k)  ||A V_k||^2 / ||A||^2       (about 1 = A has no preference; this is omega_A/||A||^2 in the log)
  wB_k     (d_out/k) ||U_k^T B||^2 / ||B||^2
  pen_k    mean over modules of (d_out/k)||U_k^T B||^2 + (d_in/k)||A V_k||^2   (the training penalty, at k)
  gradA/B  Frobenius norm of the gradient of A / B at that step (task + penalty, as the optimizer sees it)

Protocol safety (v2 observer rules): reads parameters and .grad only, never runs the model, never draws random numbers
(and restores every RNG stream anyway), does all arithmetic on CPU float64 copies. variant_finetune.py only
imports this when --obs_log is given; without it the script behaves exactly as before.
"""
import csv
import json
import math
import os
import re

import torch

from repro_utils import isolated_rng

_NAME = re.compile(r"layer\.(\d+)\.attention\.self\.(query|value)")
NAN = float("nan")


def _layer_mod(name):
    m = _NAME.search(name)
    return (int(m.group(1)), m.group(2)) if m else (-1, name)


class Observer:
    def __init__(self, model, target_modules, out_dir, ks, every=20, meta=None):
        self.every = max(1, int(every))
        self.ks = sorted({int(k) for k in ks})
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.mods = [(n, m) for n, m in model.named_modules()
                     if hasattr(m, "base_layer") and any(t in n for t in target_modules)]
        if not self.mods:
            raise RuntimeError("Observer found no LoRA q/v modules")
        kmax = max(self.ks)
        self.U, self.V, self.sig_pre, self.fro_pre = [], [], [], []
        with torch.no_grad(), isolated_rng():
            for _, m in self.mods:
                W = m.base_layer.weight.detach().float()
                if kmax > min(W.shape):
                    raise ValueError(f"k={kmax} exceeds min(W_pre.shape)={min(W.shape)}")
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)        # same call as the Tr penalty uses
                self.U.append(U[:, :kmax].cpu().double())
                self.V.append(Vh[:kmax].T.cpu().double())
                self.sig_pre.append(float(S[0]))
                self.fro_pre.append(float(W.norm()))
        layers = sorted({_layer_mod(n)[0] for n, _ in self.mods})
        self.top3 = set(layers[-3:])
        self._task_sum = self._pen_sum = None
        self._n = 0
        self._last_task = self._last_pen = None
        self._f = {}
        self._w = {}
        mod_cols = ["epoch", "step", "layer", "mod", "scale", "normA2", "normB2", "fro", "rel", "gradA", "gradB"]
        for k in self.ks:
            mod_cols += [f"R{k}", f"L{k}", f"wA{k}", f"wB{k}"]
        self.agg_cols = ["mean_rel", "top3_rel", "mean_fro"]
        for k in self.ks:
            self.agg_cols += [f"R{k}", f"L{k}", f"wA{k}", f"wB{k}", f"pen{k}", f"pen{k}_top3"]
        self._open("modules.csv", mod_cols)
        self._open("steps.csv", ["epoch", "step", "task_loss", "penalty"] + self.agg_cols)
        self._open("epochs.csv", ["epoch", "dev", "test", "asr", "train_task_loss", "train_penalty"] + self.agg_cols)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(dict(meta or {}, ks=self.ks, every=self.every, n_modules=len(self.mods)), f, indent=1)
        print(f"[OBS] logging {len(self.mods)} modules, ks={self.ks}, every={self.every} steps -> {out_dir}")

    # ---- csv plumbing
    def _open(self, name, cols):
        f = open(os.path.join(self.dir, name), "w", newline="")
        w = csv.writer(f)
        w.writerow(cols)
        f.flush()
        self._f[name], self._w[name] = f, w

    def _write(self, name, row):
        self._w[name].writerow([f"{x:.8g}" if isinstance(x, float) else x for x in row])
        self._f[name].flush()

    # ---- hooks called from the training loop (tensors are detached; no sync here)
    def accumulate_task(self, t):
        self._last_task = t
        self._task_sum = t.clone() if self._task_sum is None else self._task_sum + t
        self._n += 1

    def accumulate_pen(self, t):
        self._last_pen = t
        self._pen_sum = t.clone() if self._pen_sum is None else self._pen_sum + t

    def due(self, step, last_step):
        return step % self.every == 0 or step == last_step

    # ---- measurement
    @torch.no_grad()
    def _measure(self, epoch, step, with_grad):
        per_mod, acc = [], {c: [] for c in self.agg_cols}
        rows_rel, rows_rel_top3, rows_fro = [], [], []
        pen = {k: [] for k in self.ks}
        pen_t3 = {k: [] for k in self.ks}      # same penalty, mean over the last 3 layers only (what --tr_layers top3 trains on)
        with isolated_rng():
            for i, (name, m) in enumerate(self.mods):
                layer, mod = _layer_mod(name)
                wa, wb = m.lora_A["default"].weight, m.lora_B["default"].weight
                A, B = wa.detach().cpu().double(), wb.detach().cpu().double()
                s = float(m.scaling["default"])
                U, V = self.U[i], self.V[i]
                d_out, d_in = B.shape[0], A.shape[1]
                a2, b2 = float((A * A).sum()), float((B * B).sum())
                fro2 = s * s * float(torch.trace((B.T @ B) @ (A @ A.T)))
                if fro2 > 0:
                    Rb = torch.linalg.qr(B, mode="r").R
                    Ra = torch.linalg.qr(A.T, mode="r").R
                    sig = s * float(torch.linalg.svdvals(Rb @ Ra.T)[0])
                else:
                    sig = 0.0
                rel = sig / self.sig_pre[i]
                fro = math.sqrt(fro2)
                gA = wa.grad.detach().norm().item() if (with_grad and wa.grad is not None) else NAN
                gB = wb.grad.detach().norm().item() if (with_grad and wb.grad is not None) else NAN
                row = [epoch, step, layer, mod, s, a2, b2, fro, rel, gA, gB]
                per_k = {}
                for k in self.ks:
                    AV = A @ V[:, :k]
                    UB = U[:, :k].T @ B
                    R = s * s * float(((B @ AV) ** 2).sum()) / fro2 if fro2 > 0 else NAN
                    L = s * s * float(((UB @ A) ** 2).sum()) / fro2 if fro2 > 0 else NAN
                    wA = (d_in / k) * float((AV ** 2).sum()) / a2 if a2 > 0 else NAN
                    wB = (d_out / k) * float((UB ** 2).sum()) / b2 if b2 > 0 else NAN
                    pk = (d_in / k) * float((AV ** 2).sum()) + (d_out / k) * float((UB ** 2).sum())
                    pen[k].append(pk)
                    if layer in self.top3:
                        pen_t3[k].append(pk)
                    row += [R, L, wA, wB]
                    per_k[k] = (R, L, wA, wB)
                per_mod.append(row)
                rows_rel.append(rel)
                rows_fro.append(fro)
                if layer in self.top3:
                    rows_rel_top3.append(rel)
                for k in self.ks:
                    for j, c in enumerate(("R", "L", "wA", "wB")):
                        acc[f"{c}{k}"].append(per_k[k][j])
        mean = lambda xs: (sum(x for x in xs if not math.isnan(x)) / max(1, sum(1 for x in xs if not math.isnan(x)))) \
            if any(not math.isnan(x) for x in xs) else NAN
        agg = {"mean_rel": mean(rows_rel), "top3_rel": mean(rows_rel_top3), "mean_fro": mean(rows_fro)}
        for k in self.ks:
            for c in ("R", "L", "wA", "wB"):
                agg[f"{c}{k}"] = mean(acc[f"{c}{k}"])
            agg[f"pen{k}"] = mean(pen[k])
            agg[f"pen{k}_top3"] = mean(pen_t3[k])
        for row in per_mod:
            self._write("modules.csv", row)
        return agg

    def snapshot(self, epoch, step):
        """Call after loss.backward(), before optimizer.step(): parameters = state before this step's update."""
        agg = self._measure(epoch, step, with_grad=True)
        t = self._last_task.item() if self._last_task is not None else NAN
        p = self._last_pen.item() if self._last_pen is not None else NAN
        self._write("steps.csv", [epoch, step, t, p] + [agg[c] for c in self.agg_cols])

    def write_summary(self, info):
        """Called once after training and Pt: everything the SUMMARY line and the [Mechanism 3] lines print, plus the arguments, as summary.json."""
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)

    def epoch_end(self, epoch, dev, test, asr):
        """Call right where [EPOCH] is printed: parameters = state that dev/test/ASR were measured on."""
        agg = self._measure(epoch, "end", with_grad=False)
        tl = (self._task_sum / max(1, self._n)).item() if self._task_sum is not None else NAN
        pl = (self._pen_sum / max(1, self._n)).item() if self._pen_sum is not None else NAN
        self._write("epochs.csv", [epoch, dev, test, asr, tl, pl] + [agg[c] for c in self.agg_cols])
        self._task_sum = self._pen_sum = None
        self._n = 0
