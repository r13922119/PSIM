import random, hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from attack_utils import build_poisoned_test_dataloader

random.seed(0)
tok = AutoTokenizer.from_pretrained("roberta-large")
tok_fn = lambda ex: tok(ex["sentence"], truncation=True, max_length=None)
d = "./data/sst-2"
for split in ("train", "dev", "test"):
    ds = load_dataset("json", data_files=f"{d}/{split}.json")["train"]
    ds = ds.map(tok_fn, batched=True, remove_columns=["idx", "sentence"])
h = lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
print("python-random state before triggers:", h(random.getstate()))
dl = build_poisoned_test_dataloader(f"{d}/test.json", load_dataset, tok_fn,
        lambda e: tok.pad(e, padding="longest", return_tensors="pt"), "mn")
print("triggered-test fingerprint:", h([e["input_ids"] for e in dl.dataset]))
