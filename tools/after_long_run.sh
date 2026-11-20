#!/bin/sh
# After the long Tier A run: tools/after_long_run.sh NEW OLD DATA
# Waits for runs/f/NEW/result.json. The rule was fixed before the run ended: NEW replaces OLD as the Tier A benchmark
# model only if its loss on the body tokens of the validation text is lower (float). Only then is it quantised and run
# on the test sets (tools/finalize.py); otherwise nothing else happens and OLD stays.
cd "/Users/kixem/Desktop/LS Architecture/ls-llm"
NEW=$1
OLD=$2
DATA=$3
until [ -f runs/f/$NEW/result.json ]; do /bin/sleep 30; done
BETTER=$(.venv/bin/python -c "
import json
new, old = (json.load(open('runs/f/%s/result.json' % name))['val_body'] for name in ('$NEW', '$OLD'))
print('yes' if new < old else 'no', new, old)")
echo "$(date +%H:%M) long run $NEW against $OLD (body loss, new old): $BETTER" >> results/logs/after_long_run.txt
case "$BETTER" in
  yes*) PYTHONPATH=. .venv/bin/python tools/finalize.py ${NEW}q runs/f/$NEW/ckpt.pt $DATA --qat-steps 3000 >> results/logs/after_long_run.txt 2>&1 ;;
esac
echo "$(date +%H:%M) done" >> results/logs/after_long_run.txt
