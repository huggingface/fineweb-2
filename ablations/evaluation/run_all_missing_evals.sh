#!/bin/bash
LANGUAGES=("ru" "fr" "zh" "hi" "sw" "tr" "ar" "te" "th")
BUCKET="s3://path/to/bucket"
SCRIPT_PATH="./multilingual/launch_evals.py"

for LANG in "${LANGUAGES[@]}"; do
    GREP_PATTERN=".*gemma.*-${LANG}-.*"

    MODELS=$(aws s3 ls "$BUCKET" \
        | grep -E -- "$GREP_PATTERN" \
        | awk '{print $2}' \
        | sed 's#/$##' \
        | paste -sd ',')

    echo "$LANG: $MODELS"
    if [[ $MODELS = *[!\ ]* ]]; then

        python "$SCRIPT_PATH" \
            "$MODELS" \
            "$LANG" \
            --parallel=8 \
            --offline-datasets
    fi
done