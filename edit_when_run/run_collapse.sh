#!/bin/bash
# run_collapse.sh — reproduce ONE run that collapsed (K + Cl coord p 0.2, PZ s4, fine-tune seed 2: dev fell to 0.509 at epoch 2 in
# queue C) with the read-only observer every 5 steps, to see at which step and in which module it breaks. Protocol v2 runs are
# deterministic on the same GPU model, so the collapse should come back exactly; the observer does not change the run
# (it only reads parameters and gradients, and restores every random-number stream). Adapters go under obs_collapse/, nothing
# under adapters/ is overwritten; its log is obs_collapse/collapse_<date>.log (not in logs/). One card, can share it with the run_psel3.sh queues. About 20-40 min.
#   cd ~/psim_proj && nohup bash run_collapse.sh > /dev/null 2>&1 &
NAME=collapse; source ./v2_common.sh
# keep this run's log OUT of logs/: it has the same experiment name as the queue-C run it reproduces, and the analysis scripts read
# every logs/*.log (if the two ever differed, ft_logs.pick would stop with "completed twice with different numbers")
mkdir -p obs_collapse; LOGFILE="obs_collapse/collapse_${DATE}.log"; LOGPARSEDFILE="obs_collapse/collapse_${DATE}_parsed.log"
run variant_finetune.py "collapse repro s4 wh0.8 p0.2 ft2 cl lr2e-4 obs5" \
  --pz_path pz_epoch/n1500_s4_ep2/pytorch_model.bin --wpre_file pz_wpre/n1500_s4_ep2_whiten_r0.8.json --seed 2 --lr 2e-4 \
  --use_spectral_rescaling --use_pretrained_dropout --cl_dropout_p 0.2 --obs_log obs_collapse --obs_every 5
finish