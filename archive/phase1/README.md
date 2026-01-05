# Phase 1: Entropy側DWT + LL Prior

## 概要
最小改変でWeConveneの核心機能を統合。
- Entropy側でDWT適用
- LL→LH→HL→HH順序でエンコード  
- HFバンドにLL priorを注入

## 使い方

### 基本学習（選択肢3なし）
```bash
cd /workspace/LIC-HPCM-WeConvene/archive/phase1

python train_phase1.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --batch-size 16 \
  --epochs 3001 \
  --save_path ./checkpoints
```

### 選択肢3あり（LL=10ステップ、HF=4ステップ）
```bash
python train_phase1.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --skip_s3_for_hf \
  --save_path ./checkpoints_option3
```

## パラメータ

- `--skip_s3_for_hf`: 選択肢3を有効化（HFバンドのs3処理をスキップ）
- `--lambda`: Rate-distortion tradeoff (0.013, 0.025, 0.05, 0.1)
- `--batch-size`: バッチサイズ（デフォルト: 32）
- `--epochs`: エポック数（デフォルト: 3001）

## 期待される効果

| 設定 | BD-rate改善 | HF計算量 |
|------|-----------|---------|
| 選択肢3なし | -2~-5% | 100% |
| 選択肢3あり | -2~-5% | 40% |

## 次のステップ

BD-rate改善が確認できたら Phase 2 へ進む。
