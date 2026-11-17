#!/usr/bin/env python3
"""margin_eval.py — paired margin decomposition on SST-2 test (evaluation only: no training, nothing existing is modified).

What it scores (50 configurations): for each of the 5 panel PZs in --choice
  the PZ itself before fine-tuning (its own classification head, no LoRA update)            5
  the last-epoch adapter adapters/<name>/last/ of plain LoRA, cl0.2, clsvd0.2, seeds 0-2    45
on the same three sets of sentences: clean positive, the same positives with "mn" inserted, clean negative.
Not included on purpose: clsvd0.3, the truncated (W_pre,K) arms, head swapping, rescaling of the update.

How (everything that touches the model mirrors existing code, so the numbers can be checked against the logs)
  data      load_dataset + tokenizer(sentence, truncation=True, max_length=None) + tokenizer.pad(longest), batch 32, no shuffle,
            the whole test file in file order: the test_dataloader of variant_finetune.py.
  trigger   attack_utils.build_poisoned_test_dataloader (random.Random(PLACEMENT_SEED = 1234)): the loader behind every logged ASR.
            The placement table (which word index got "mn") is rebuilt with attack_utils.insert_trigger and checked token by
            token against that loader, so the per-sentence ids in the output are the sentences that were actually scored.
  model     roberta-large + the PZ's state dict, built ONCE per PZ. The PZ is scored first; then PeftModel.from_pretrained
            wraps it once and every adapter is put in place with load_adapter(dir, "default") (weights only, same adapter
            slot), each time followed by a bit-for-bit read-back of all 100 tensors (48 lora_A, 48 lora_B, 4 classifier).
  state     model.eval(), no_grad, float32, the deterministic settings of variant_finetune.py, the Ada (assert_gpu).
  numbers   m = logit_1 - logit_0; definitions in margin_metrics.py.

Two checks, each stops the run (exit 1) the moment it fails; a failed configuration's .npz is still on disk for inspection:
  1. the PZ before fine-tuning: its scores against logs/scores/dose_n<n>_s<seed>/ep<epoch>.npz (down_pos, down_neg, down_asr):
     largest |difference| <= --tol (1e-4) and no sentence's prediction changes. (The stored clean scores were batched per
     class, here per file, so the padding differs; the triggered scores use the same batches.)
  2. every adapter: test accuracy and ASR, printed with 4 decimals, must equal last_test_acc and last_ASR of its own
     training log (the last epoch, before Pt).
The csv summaries are written only after all 50 configurations passed.

Output (new files only; refuses to overwrite unless --overwrite), under --out_dir (default logs/margin_decomp):
  npz/<name>.npz              per sentence: idx, the two logits, m(x), m(x + mn), s(x), insert_pos  (margin_metrics.save_scores)
  placement.csv               idx, n_words, insert_pos, the clean and the triggered sentence
  sanity_checks.csv           one row per configuration: which check, what was compared, pass or fail
  summary_checkpoints.csv, summary_pz.csv, counterfactual_checkpoints.csv, counterfactual_pz.csv   (margin_metrics.write_summaries)

  python margin_eval.py --dry      no GPU, loads no model: are the 5 PZs, 45 adapters, their meta.json, the logged numbers and
                                   the stored PZ scores all there and consistent?
  python margin_eval.py            the evaluation (about 50 x (1821 + 909) short sentences; one card)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before torch / any CUDA context, as in variant_finetune.py
