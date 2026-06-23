#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_lopo.sh — Roda todos os folds LOPO para um dataset/config
#
# Uso:
#   bash scripts/run_lopo.sh --config configs/experiments/minds_2nd_resnet18.yaml
#   bash scripts/run_lopo.sh -d minds -s 2nd -m resnet18 -im Skeleton-DML -r 63
#   bash scripts/run_lopo.sh -d ksl -m resnet18 -im Skeleton-DML -r 15
#
# Para KSL (folds de grupo):
#   bash scripts/run_lopo.sh -d ksl -r 15 --folds "0,1,2,3|4,5,6,7|8,9,10,11|12,13,14,15|16,17,18,19"
#
# Para Include50 (split fixo: val=1, test=2):
#   bash scripts/run_lopo.sh -d include50 -r 15 --fixed-split
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG=""
DATASET=""
SUBSET="2nd"
MODEL="resnet18"
IMAGE_METHOD="Skeleton-DML"
LR="0.0001"
WD="0.0001"
REF="01"
FOLDS=""
FIXED_SPLIT=false
EXTRA_ARGS=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)     CONFIG="$2";       shift 2 ;;
    -d|--dataset) DATASET="$2";      shift 2 ;;
    -s|--subset)  SUBSET="$2";       shift 2 ;;
    -m|--model)   MODEL="$2";        shift 2 ;;
    -im)          IMAGE_METHOD="$2"; shift 2 ;;
    -lr)          LR="$2";           shift 2 ;;
    -wd)          WD="$2";           shift 2 ;;
    -r|--ref)     REF="$2";          shift 2 ;;
    --folds)      FOLDS="$2";        shift 2 ;;
    --fixed-split) FIXED_SPLIT=true; shift ;;
    --no-impute)  EXTRA_ARGS="$EXTRA_ARGS --no-impute"; shift ;;
    *) shift ;;
  esac
done

# ── Resolve dataset from YAML if provided ─────────────────────────────────────
if [[ -n "$CONFIG" ]]; then
  DATASET=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('dataset',''))" 2>/dev/null || echo "$DATASET")
  SUBSET=$(python3  -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('subset','2nd'))" 2>/dev/null || echo "$SUBSET")
  MODEL=$(python3   -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('model','resnet18'))" 2>/dev/null || echo "$MODEL")
  IMAGE_METHOD=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('image_method','Skeleton-DML'))" 2>/dev/null || echo "$IMAGE_METHOD")
  LR=$(python3  -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('learning_rate',0.0001))" 2>/dev/null || echo "$LR")
  WD=$(python3  -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('weight_decay',0.0001))" 2>/dev/null || echo "$WD")
fi

if [[ -z "$DATASET" ]]; then
  echo "Erro: informe --dataset ou --config"; exit 1
fi

BASE_CMD="python scripts/03_train.py -d $DATASET -s $SUBSET -m $MODEL -im $IMAGE_METHOD -lr $LR -wd $WD -r $REF $EXTRA_ARGS"
[[ -n "$CONFIG" ]] && BASE_CMD="$BASE_CMD --config $CONFIG"

LOG_DIR="experiments/$REF/$DATASET/logs"
mkdir -p "$LOG_DIR"

echo "══════════════════════════════════════════════════════════"
echo "  Dataset:  $DATASET | Subset: $SUBSET"
echo "  Model:    $MODEL   | Rep:    $IMAGE_METHOD"
echo "  LR: $LR | WD: $WD | Ref: $REF"
echo "══════════════════════════════════════════════════════════"

# ── LOPO por dataset ──────────────────────────────────────────────────────────

run_fold() {
  local vp="$1" tp="$2" fold_id="$3"
  local log="$LOG_DIR/fold_${fold_id}.log"
  echo ""
  echo "▶ Fold $fold_id — val=$vp  test=$tp"
  echo "  Log → $log"
  $BASE_CMD -vp "$vp" -tp "$tp" 2>&1 | tee "$log"
  echo "✓ Fold $fold_id concluído"
}

case "$DATASET" in

  minds)
    # 12 signers: test=N, val=N+1 (circular)
    SIGNERS=(1 2 3 4 5 6 7 8 9 10 11 12)
    N=${#SIGNERS[@]}
    for i in "${!SIGNERS[@]}"; do
      TP=${SIGNERS[$i]}
      VP=${SIGNERS[$(( (i + 1) % N ))]}
      run_fold "$VP" "$TP" "$TP"
    done
    ;;

  ufop)
    # 5 signers: pares (vp, tp) usados no paper
    PAIRS=("1,2" "2,3" "3,4" "4,5" "5,1")
    for pair in "${PAIRS[@]}"; do
      VP=$(echo "$pair" | cut -d, -f1)
      TP=$(echo "$pair" | cut -d, -f2)
      run_fold "$VP" "$TP" "${VP}_${TP}"
    done
    ;;

  ksl)
    # 5 grupos de 4 pessoas cada
    if [[ -n "$FOLDS" ]]; then
      IFS='|' read -ra FOLD_LIST <<< "$FOLDS"
    else
      FOLD_LIST=("0,1,2,3" "4,5,6,7" "8,9,10,11" "12,13,14,15" "16,17,18,19")
    fi
    for i in "${!FOLD_LIST[@]}"; do
      GROUP="${FOLD_LIST[$i]}"
      run_fold "$GROUP" "$GROUP" "$i"
    done
    ;;

  include50)
    # Split fixo: person 0=train, 1=val, 2=test
    run_fold "1" "2" "official_split"
    ;;

  *)
    # Dataset customizado: usa --folds "vp1,tp1|vp2,tp2|..."
    if [[ -z "$FOLDS" ]]; then
      echo "Dataset '$DATASET' desconhecido. Use --folds 'vp1,tp1|vp2,tp2|...'"
      exit 1
    fi
    IFS='|' read -ra FOLD_LIST <<< "$FOLDS"
    for i in "${!FOLD_LIST[@]}"; do
      VP=$(echo "${FOLD_LIST[$i]}" | cut -d, -f1)
      TP=$(echo "${FOLD_LIST[$i]}" | cut -d, -f2)
      run_fold "$VP" "$TP" "$i"
    done
    ;;
esac

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Todos os folds concluídos."
echo "  Resultados → experiments/$REF/$DATASET/"
echo "  python scripts/04_show_results.py -r $REF -d $DATASET"
echo "══════════════════════════════════════════════════════════"
