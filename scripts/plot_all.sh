#!/bin/bash

# Setup: pip install pandas matplotlib

# Script to generate all ASR leaderboard plots
# Usage: 
#   ./scripts/plot_all.sh
#   ./scripts/plot_all.sh --highlight "model_name"
#   ./scripts/plot_all.sh --en-shortform-csv custom.csv
#   ./scripts/plot_all.sh --custom-model "MY MODEL" --model-size 2.0 \
#       --en-shortform-wer 5.2 --en-shortform-rtfx 500 \
#       --multilingual-wer 6.5 --multilingual-rtfx 800 \
#       --en-longform-wer 10.5 --en-longform-rtfx 1000

# Default values
EN_SHORTFORM_CSV="scripts/data/en_shortform.csv"
MULTILINGUAL_CSV="scripts/data/multilingual.csv"
EN_LONGFORM_CSV="scripts/data/en_longform.csv"
HIGHLIGHT_MODEL=""
CUSTOM_MODEL_NAME=""
MODEL_SIZE=""
EN_SHORTFORM_WER=""
EN_SHORTFORM_RTFX=""
MULTILINGUAL_WER=""
MULTILINGUAL_RTFX=""
EN_LONGFORM_WER=""
EN_LONGFORM_RTFX=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --en-shortform-csv)
            EN_SHORTFORM_CSV="$2"
            shift 2
            ;;
        --multilingual-csv)
            MULTILINGUAL_CSV="$2"
            shift 2
            ;;
        --en-longform-csv)
            EN_LONGFORM_CSV="$2"
            shift 2
            ;;
        --highlight)
            HIGHLIGHT_MODEL="$2"
            shift 2
            ;;
        --custom-model)
            CUSTOM_MODEL_NAME="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --en-shortform-wer)
            EN_SHORTFORM_WER="$2"
            shift 2
            ;;
        --en-shortform-rtfx)
            EN_SHORTFORM_RTFX="$2"
            shift 2
            ;;
        --multilingual-wer)
            MULTILINGUAL_WER="$2"
            shift 2
            ;;
        --multilingual-rtfx)
            MULTILINGUAL_RTFX="$2"
            shift 2
            ;;
        --en-longform-wer)
            EN_LONGFORM_WER="$2"
            shift 2
            ;;
        --en-longform-rtfx)
            EN_LONGFORM_RTFX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --en-shortform-csv PATH    CSV file for shortform (default: scripts/data/en_shortform.csv)"
            echo "  --multilingual-csv PATH    CSV file for multilingual (default: scripts/data/multilingual.csv)"
            echo "  --en-longform-csv PATH     CSV file for longform (default: scripts/data/en_longform.csv)"
            echo "  --highlight MODEL          Model name to highlight"
            echo "  --custom-model NAME        Custom model name"
            echo "  --model-size SIZE          Model size in billions"
            echo "  --en-shortform-wer WER     Shortform WER"
            echo "  --en-shortform-rtfx RTFX   Shortform RTFx"
            echo "  --multilingual-wer WER     Multilingual WER"
            echo "  --multilingual-rtfx RTFX   Multilingual RTFx"
            echo "  --en-longform-wer WER      Longform WER"
            echo "  --en-longform-rtfx RTFX    Longform RTFx"
            exit 1
            ;;
    esac
done

# Validate custom model arguments
if [[ -n "$CUSTOM_MODEL_NAME" ]]; then
    if [[ -z "$MODEL_SIZE" ]]; then
        echo "Error: When --custom-model is specified, --model-size must be provided"
        exit 1
    fi
fi

# Check which tasks have complete custom model data
HAS_SHORTFORM_DATA=false
HAS_MULTILINGUAL_DATA=false
HAS_LONGFORM_DATA=false

if [[ -n "$CUSTOM_MODEL_NAME" && -n "$MODEL_SIZE" ]]; then
    if [[ -n "$EN_SHORTFORM_WER" && -n "$EN_SHORTFORM_RTFX" ]]; then
        HAS_SHORTFORM_DATA=true
    fi
    if [[ -n "$MULTILINGUAL_WER" && -n "$MULTILINGUAL_RTFX" ]]; then
        HAS_MULTILINGUAL_DATA=true
    fi
    if [[ -n "$EN_LONGFORM_WER" && -n "$EN_LONGFORM_RTFX" ]]; then
        HAS_LONGFORM_DATA=true
    fi
fi

# Build command-line arguments for each script
SHORTFORM_ARGS=""
MULTILINGUAL_ARGS=""
LONGFORM_ARGS=""

# Determine prefix for output filenames (sanitize for filesystem)
OUTPUT_PREFIX=""
if [[ -n "$CUSTOM_MODEL_NAME" ]]; then
    # Use custom model name as prefix
    OUTPUT_PREFIX=$(echo "$CUSTOM_MODEL_NAME" | sed 's/[\/: ]/_/g' | sed 's/__*/_/g')
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_"
elif [[ -n "$HIGHLIGHT_MODEL" ]]; then
    # Use highlight model name as prefix
    OUTPUT_PREFIX=$(echo "$HIGHLIGHT_MODEL" | sed 's/[\/: ]/_/g' | sed 's/__*/_/g')
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_"
fi

if [[ -n "$HIGHLIGHT_MODEL" ]]; then
    SHORTFORM_ARGS="--highlight \"$HIGHLIGHT_MODEL\""
    MULTILINGUAL_ARGS="--highlight \"$HIGHLIGHT_MODEL\""
    LONGFORM_ARGS="--highlight \"$HIGHLIGHT_MODEL\""
fi

if [[ "$HAS_SHORTFORM_DATA" == true ]]; then
    SHORTFORM_ARGS="$SHORTFORM_ARGS --custom-model \"$CUSTOM_MODEL_NAME,$EN_SHORTFORM_WER,$EN_SHORTFORM_RTFX,$MODEL_SIZE\""
    echo "Custom model will be added to EN Shortform plots"
fi

if [[ "$HAS_MULTILINGUAL_DATA" == true ]]; then
    MULTILINGUAL_ARGS="$MULTILINGUAL_ARGS --custom-model \"$CUSTOM_MODEL_NAME,$MULTILINGUAL_WER,$MULTILINGUAL_RTFX,$MODEL_SIZE\""
    echo "Custom model will be added to Multilingual plots"
fi

if [[ "$HAS_LONGFORM_DATA" == true ]]; then
    LONGFORM_ARGS="$LONGFORM_ARGS --custom-model \"$CUSTOM_MODEL_NAME,$EN_LONGFORM_WER,$EN_LONGFORM_RTFX,$MODEL_SIZE\""
    echo "Custom model will be added to EN Longform plots"
fi

# Add output filename arguments with prefix if applicable
if [[ -n "$OUTPUT_PREFIX" ]]; then
    SHORTFORM_ARGS="$SHORTFORM_ARGS --rtfx-output \"${OUTPUT_PREFIX}en_shortform_rtfx_wer.png\" --size-output \"${OUTPUT_PREFIX}en_shortform_size_wer.png\""
    MULTILINGUAL_ARGS="$MULTILINGUAL_ARGS --rtfx-output \"${OUTPUT_PREFIX}multilingual_rtfx_wer.png\" --size-output \"${OUTPUT_PREFIX}multilingual_size_wer.png\""
    LONGFORM_ARGS="$LONGFORM_ARGS --rtfx-output \"${OUTPUT_PREFIX}longform_rtfx_wer.png\" --size-output \"${OUTPUT_PREFIX}longform_size_wer.png\""
    echo "Output plots will be prefixed with: ${OUTPUT_PREFIX}"
fi

# Add blank line before plot generation if any options were specified
if [[ -n "$HIGHLIGHT_MODEL" || -n "$CUSTOM_MODEL_NAME" ]]; then
    echo ""
fi

# Run the plotting scripts
FAILED_PLOTS=()
SKIPPED_PLOTS=()

# Only generate plots if: no custom model specified OR custom model has complete data for that task
SHOULD_RUN_SHORTFORM=true
SHOULD_RUN_MULTILINGUAL=true
SHOULD_RUN_LONGFORM=true

if [[ -n "$CUSTOM_MODEL_NAME" ]]; then
    # Custom model specified - only run tasks with complete data
    if [[ "$HAS_SHORTFORM_DATA" == false ]]; then
        SHOULD_RUN_SHORTFORM=false
    fi
    if [[ "$HAS_MULTILINGUAL_DATA" == false ]]; then
        SHOULD_RUN_MULTILINGUAL=false
    fi
    if [[ "$HAS_LONGFORM_DATA" == false ]]; then
        SHOULD_RUN_LONGFORM=false
    fi
fi

if [[ "$SHOULD_RUN_SHORTFORM" == true ]]; then
    echo "Generating EN Shortform plots..."
    if eval python scripts/plot_en_shortform.py --csv_file \"$EN_SHORTFORM_CSV\" $SHORTFORM_ARGS; then
        echo "✓ EN Shortform plots generated successfully"
    else
        echo "✗ EN Shortform plots failed (model not found or other error)"
        FAILED_PLOTS+=("EN Shortform")
    fi
    echo ""
else
    echo "Skipping EN Shortform plots (custom model specified but no shortform metrics provided)"
    SKIPPED_PLOTS+=("EN Shortform")
    echo ""
fi

if [[ "$SHOULD_RUN_MULTILINGUAL" == true ]]; then
    echo "Generating Multilingual plots..."
    if eval python scripts/plot_multilingual.py --csv_file \"$MULTILINGUAL_CSV\" $MULTILINGUAL_ARGS; then
        echo "✓ Multilingual plots generated successfully"
    else
        echo "✗ Multilingual plots failed (model not found or other error)"
        FAILED_PLOTS+=("Multilingual")
    fi
    echo ""
else
    echo "Skipping Multilingual plots (custom model specified but no multilingual metrics provided)"
    SKIPPED_PLOTS+=("Multilingual")
    echo ""
fi

if [[ "$SHOULD_RUN_LONGFORM" == true ]]; then
    echo "Generating EN Longform plots..."
    if eval python scripts/plot_en_longform.py --csv_file \"$EN_LONGFORM_CSV\" $LONGFORM_ARGS; then
        echo "✓ EN Longform plots generated successfully"
    else
        echo "✗ EN Longform plots failed (model not found or other error)"
        FAILED_PLOTS+=("EN Longform")
    fi
    echo ""
else
    echo "Skipping EN Longform plots (custom model specified but no longform metrics provided)"
    SKIPPED_PLOTS+=("EN Longform")
    echo ""
fi

if [[ ${#FAILED_PLOTS[@]} -eq 0 ]]; then
    if [[ ${#SKIPPED_PLOTS[@]} -eq 0 ]]; then
        echo "Done! All plots generated successfully."
    else
        echo "Done! Generated plots successfully. Skipped: ${SKIPPED_PLOTS[*]}"
    fi
else
    echo "Done with warnings. Failed plots: ${FAILED_PLOTS[*]}"
    if [[ ${#SKIPPED_PLOTS[@]} -gt 0 ]]; then
        echo "Skipped plots: ${SKIPPED_PLOTS[*]}"
    fi
    exit 1
fi
