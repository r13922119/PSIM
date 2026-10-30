#!/bin/bash
# setup_proj_dir.sh <repo folder> <new folder> <folder holding the 10/30 files>
# A second working folder for the --tr_proj runs while queues that use the OLD variant_finetune.py are still running in the repo.
# Everything in the new folder is a link to the repo (data, pz*, pz_wpre, logs, adapters, v2_common.sh, ...), so logs and adapters
# still land in the repo, except three real copies of the 10/30 versions: variant_finetune.py, run_psel.sh, ft_logs.py.
# Each queue computes its code hash from the variant_finetune.py in ITS folder, so the two folders never mix their records.
# After the old queues finish, copy the three files into the repo as usual and delete the new folder (links only; nothing is lost).
set -eu
[ $# -eq 3 ] || { echo "usage: bash setup_proj_dir.sh <repo> <new folder> <folder with the 10/30 files>"; exit 1; }
REPO=$(realpath "$1"); NEW="$2"; SRC=$(realpath "$3")
for f in variant_finetune.py run_psel.sh ft_logs.py; do [ -f "$SRC/$f" ] || { echo "$SRC/$f missing"; exit 1; }; done
grep -q -- "--tr_proj" "$SRC/variant_finetune.py" || { echo "$SRC/variant_finetune.py is not the 10/30 version (no --tr_proj)"; exit 1; }
mkdir -p "$NEW"; cd "$NEW"
for f in "$REPO"/*; do ln -sfn "$f" "$(basename "$f")"; done
for f in variant_finetune.py run_psel.sh ft_logs.py; do rm -f "$f"; cp "$SRC/$f" "$f"; done
echo "ready: $(pwd) — variant_finetune.py, run_psel.sh, ft_logs.py are copies; everything else links to $REPO"