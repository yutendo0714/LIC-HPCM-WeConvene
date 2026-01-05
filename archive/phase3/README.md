# Phase 3: Full Wavelet Integration

## 概要
完全wavelet-domain化。
- Phase 2の機能（DWT + LL prior + WeConv）
- h_a/h_sもwavelet-domain化
- 周波数適応的hyper-prior生成

## 前提条件
Phase 2で学習済みチェックポイントが必要（h_a/h_s以外を転移学習）

## 使い方

### 基本学習
```bash
cd /workspace/LIC-HPCM-WeConvene/archive/phase3

python train_phase3.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --use_simple_weconv \
  --batch-size 8 \
  --save_path ./checkpoints
```

### 選択肢3も併用
```bash
python train_phase3.py \
  --train_dataset /path/to/train \
  --test_dataset /path/to/test \
  --lambda 0.013 \
  --use_simple_weconv \
  --skip_s3_for_hf \
  --save_path ./checkpoints_option3
```

## Phase 2からの転移学習（TODO）

```python
# train_phase3.py に以下を追加予定
# --pretrain /path/to/phase2_checkpoint.pth
# h_a/h_s以外の重みをロード（g_a/g_s/HPCM components）
```

## 2段階学習戦略

```bash
# Stage 1: h_a/h_sのみ学習（20 epochs）
python train_phase3.py \
  --freeze_except h_a h_s \
  --epochs 20 \
  --lr 1e-4

# Stage 2: 全体をfine-tune（30 epochs）
python train_phase3.py \
  --pretrain ./checkpoints/epoch_20.pth \
  --epochs 50 \
  --lr 1e-5
```

## パラメータ

- `--use_simple_weconv`: Simple版WeConv使用（推奨）
- `--skip_s3_for_hf`: 選択肢3を有効化
- `--lambda`: Rate-distortion tradeoff
- `--batch-size`: 8推奨（メモリ使用量大）

## 期待される効果

| 設定 | BD-rate改善 | パラメータ | 学習時間 |
|------|-----------|----------|---------|
| Phase 2比 | -2~-5% | +7% | +30% |
| Phase 1比 | -10~-15% | +11% | +50% |

## 理論的一貫性

全パイプラインでwavelet-aware:
- ✅ Feature extraction (g_a): Wavelet
- ✅ Hyper-prior (h_a/h_s): Wavelet
- ✅ Entropy coding: Wavelet
- ✅ Reconstruction (g_s): Wavelet

## トラブルシューティング

### メモリ不足
- `--batch-size 4` に削減
- `--use_simple_weconv` 使用

### 学習不安定
- 学習率を下げる: `--lr 5e-5`
- Gradient clipping強化
- Warmup追加

### Phase 2より悪化
- h_a/h_sの学習エポックを増やす
- 2段階学習を試す
- Phase 2に留まる判断も
