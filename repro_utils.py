"""Shared helpers for the frozen "observer protocol" (v2).

Trainer side (global RNG, left exactly as in PSIM): train/dev data poisoning, model init,
the training loop, the per-epoch dev pass, best-dev selection and checkpoint saving.

Observer side (isolated RNG, must never move any trainer stream): test accuracy, ASR,
trigger placement for the test set, and every diagnostic.
"""
import hashlib
import json
import os
import platform
import random
from contextlib import contextmanager
from importlib import metadata

import numpy as np
import torch

PROTOCOL = "v2"
EXPECTED_GPU_DEFAULT = "NVIDIA RTX 6000 Ada Generation"
PLACEMENT_SEED = 1234   # fixed seed for test-set trigger placement (observer randomness)


def assert_env():
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("export PYTHONNOUSERSITE=1 before launching python "
                           "(user-site packages in ~/.local shadow the conda env).")


def assert_gpu():
    expected = os.environ.get("EXPECTED_GPU", EXPECTED_GPU_DEFAULT)
    got = torch.cuda.get_device_name(0)
    if got != expected:
        raise RuntimeError(f"GPU is '{got}' but protocol {PROTOCOL} requires '{expected}'. "
                           "Dropout masks differ between GPU models, so results are not comparable.")


@contextmanager
def isolated_rng():
    """Run observer code without moving any RNG stream (python, numpy, torch CPU, torch CUDA)."""
    py, npy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def observer(model):
    """Use for EVERY observer/debug call that runs the model.
    Isolates all RNG streams AND restores model.training, and disables grad.
    (isolated_rng alone does not undo model.eval() left behind by an eval function.
    That only matters if the training loop does not call model.train() itself at the start of
    every epoch; if it does, the leak is harmless, and this is a cheap safety net.)"""
    was_training = model.training
    try:
        with isolated_rng(), torch.no_grad():
            yield
    finally:
        model.train(was_training)


def rng_fp():
    """Short fingerprint of every RNG stream. Consumes nothing."""
    h = hashlib.sha1()
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for s in torch.cuda.get_rng_state_all():
            h.update(s.numpy().tobytes())
    h.update(repr(random.getstate()).encode())
    h.update(np.random.get_state()[1].tobytes())
    return h.hexdigest()[:8]


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ver(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "-"


def env_info(script_path, files=None):
    info = {"protocol": PROTOCOL, "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda}
    for p in ("transformers", "tokenizers", "datasets", "peft", "numpy", "huggingface_hub", "accelerate"):
        info[p] = _ver(p)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", "?"))
    info["script_sha256"] = file_sha256(script_path)[:12]
    for label, p in (files or {}).items():
        info[f"{label}_sha256"] = file_sha256(p)[:12]
    return info


def log_env(script_path, files=None):
    info = env_info(script_path, files)
    print("[ENV] " + json.dumps(info, sort_keys=True))
    return info


def write_meta(meta_path, info, ckpt_path, **fields):
    meta = dict(info)
    meta.update(fields)
    meta["checkpoint_sha256"] = file_sha256(ckpt_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print(f"[META] wrote {meta_path}")