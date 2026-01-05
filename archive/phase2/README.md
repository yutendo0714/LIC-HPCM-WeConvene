# Phase 2: WeConv Block Integration

## 概要
Phase 1にWeConvブロックを追加。
- Phase 1の機能（DWT + LL prior）
- g_a/g_sにWeConvブロック追加（各2箇所）
- Wavelet-domain特徴処理

## 前提条件
Phase 1で学習済みチェックポイントが必要（転移学習推奨）

## 使い方

### Simple版（推奨）
```bash
cd /workspace/LIC-HPCM-WeConvene/archive/phase2

python train_phase2.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --use_simple_weconv \
  --batch-size 16 \
  --save_path ./checkpoints
```

### Full版（より強力だが重い）
```bash
python train_phase2.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --batch-size 8 \
  --save_path ./checkpoints_full
```

### 選択肢3も併用
```bash
python train_phase2.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --use_simple_weconv \
  --skip_s3_for_hf \
  --save_path ./checkpoints_option3
```

## Phase 1からの転移学習（TODO）

```python
# train_phase2.py に以下を追加予定
# --pretrain /path/to/phase1_checkpoint.pth
# g_a/g_s以外の重みをロード
```

## パラメータ

- `--use_simple_weconv`: Simple版WeConv使用（デフォルト: True）
- `--skip_s3_for_hf`: 選択肢3を有効化
- `--lambda`: Rate-distortion tradeoff
- `--batch-size`: Simple版=16、Full版=8推奨

## 期待される効果

| 設定 | BD-rate改善 | パラメータ | メモリ |
|------|-----------|----------|--------|
| Simple版 | -5~-8% | +4% | +10% |
| Full版 | -7~-10% | +4% | +20% |
| +選択肢3 | 同上 | 同上 | -15% |

## 次のステップ

大幅改善（-10%以上）が確認できたら Phase 3 へ進む。
