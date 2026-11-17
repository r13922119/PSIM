"""CPU end-to-end test of margin_eval.py on a miniature model (needs torch, transformers, peft, datasets; no GPU, no download).
Run from the PSIM folder:   PYTHONNOUSERSITE=1 python test_margin_e2e.py          (about a minute on CPU)

It cannot check the real numbers (that is what margin_eval.py's own two checks do on the Ada). It checks that the script's
control flow and bookkeeping are right with the installed library versions, before any GPU time is spent:
the adapter switching in place, the data pipeline, the two checks, the files written, and that a wrong input stops the run.

Everything is built in a temporary folder and removed at the end; the repo is only read (data/sst-2/test.json, and the
modules attack_utils, bias_probe, repro_utils, ft_logs are imported). The miniature copy of what margin_eval.py reads:
  model     a 24-layer RoBERTa with hidden size 32 and a word-level tokenizer, saved as the --hf folder (48 q/v modules, like roberta-large)
  PZs       5 state dicts + meta.json (repro_utils.write_meta, the fields poisoned_pretrain.py writes)
  adapters  45 last/ folders (plain, cl0.2, clsvd0.2 x seeds 0-2 per PZ), written with the calls variant_finetune.py uses:
            get_peft_model(LoraConfig(SEQ_CLS, r 8, alpha 16, lora_dropout 0.1, query/value)), save_pretrained, repro_utils.write_meta
  data      the first 160 rows of the real SST-2 test file
  stored PZ scores   produced with bias_probe.class_loaders / class_probe / _asr, the way sst2dev_eval.py produced the real ones
  training logs      variant_finetune.py's line format; the last-epoch numbers come from an INDEPENDENT evaluation: a fresh
                     model per adapter + PeftModel.from_pretrained, variant_finetune.evaluate_accuracy's loop, attack_utils.compute_asr

Checks
  1. --dry accepts the miniature inputs; with one adapter's meta.json claiming another seed it lists the problem and evaluates nothing;
  2. the full run passes its 5 + 45 checks and writes 50 score files, placement.csv, sanity_checks.csv and the four summaries;
  3. every adapter's per-sentence scores from the in-place switching equal those of a freshly built model with that adapter
     loaded alone (so no weight of a previous adapter or of the PZ head is left behind), and the PZ's equal the plain model;
  4. placement.csv equals margin_metrics.build_placement with attack_utils.insert_trigger, and idx / insert_pos in the score files match it;
  5. a second run into the same folder is refused without --overwrite;
  6. a training log that is one sentence off stops the run at that adapter (exit 1) and no summary is written;
  7. stored PZ scores that are 2e-4 off stop the run at that PZ (exit 1) before any adapter is evaluated.
"""
import contextlib, csv, io, json, os, shutil, sys, tempfile, time
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # nothing here may reach the network

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_utils, bias_probe, margin_eval, margin_metrics as M, repro_utils
from attack_utils import build_poisoned_test_dataloader, compute_asr
