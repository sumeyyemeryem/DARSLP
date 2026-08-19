# DARSLP

**Disentangle and Regularize: Sign Language Production with Articulator-Based
Disentanglement and Channel-Aware Regularization**

Taşyürek et al., WACV 2026.

DARSLP is a two-stage, non-autoregressive Sign Language Production pipeline:

1. **Stage 1 — `DisentangledAE`**: a pose autoencoder that encodes each frame
   into four separate latent subspaces (upper body, right hand, left hand,
   face) instead of one unified latent vector.
2. **Stage 2 — `DARSLPGenerator`**: a non-autoregressive transformer that
   predicts those four latent codes directly from text (BERT embeddings),
   optionally regularized in a second training phase by channel-aware KL
   priors computed from the training set's latent distribution.

A non-disentangled `StandardAE` / `StandardGenerator` pair is included as the
ablation baseline referenced in the paper.

## Repository layout

```
configs/                 Stage 2 model/training hyperparameters (per dataset)
data/
  reference_pose.pt       static decoder query pose (DARSLPGenerator)
  channel_priors/          precomputed per-channel mean/std priors (Phase 2 KL)
models/                   pretrained Stage 1 AE checkpoints (.pth)
src/
  data/SLPDataset.py       PyTorch Dataset (text embeddings + poses + encodings)
  models/                  DisentangledAE, StandardAE, DARSLPGenerator, StandardGenerator
  loss/channel_kl_loss.py  Phase 2 channel-aware KL regularization
  KL-div/                  scripts to compute channel_priors from a trained AE
  prediction/              inference (predict_*.py) and back-translation export (export_*.py)
  visualization/           render predicted pose sequences as PNG/MP4
  utils/                   config loading, masking, pixel back-translation
precompute_text_embeddings.py   Stage 1/2 prerequisite: per-sample BERT embeddings
precompute_encodings.py         Stage 2 prerequisite: per-sample AE latents
train_stage1.py                 Stage 1 (AE) training
train_stage2.py                 Stage 2 (generator) training
```

## Installation

Tested with Python 3.10, CUDA 12.4.

```bash
# torch/torchvision ship CUDA-specific wheels — install for your platform first
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data

`SLPDataset` (`src/data/SLPDataset.py`) expects a `.pt` file per split
(train/dev/test), containing a dict keyed by sample ID:

```python
{
  "<sample_id>": {
      "text":     str,               # spoken-language sentence
      "gloss":    str,
      "speaker":  str,
      "poses_3d": Tensor[F, 178, 3],  # 3D pose sequence, see joint layout below
  },
  ...
}
```

Joint layout (178 points): `0–7` upper body, `8–28` right hand, `29–49` left
hand, `50–177` face (128 landmarks) — see `src/visualization/keypoint_def.py`
for the named mapping and `src/prediction/predict_phoenix.py` / `predict_csl.py`
for the full index breakdown.

This repository does **not** include a raw-corpus-to-`.pt` conversion script:
building `train.pt` / `dev.pt` / `test.pt` from the official
[PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
and [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/) releases
is dataset-specific preprocessing (pose extraction + alignment) that depends
on each dataset's own license and distribution terms. Once you have a `.pt`
file in the format above, everything downstream (text embeddings, AE
training, generator training, inference) runs the same way regardless of
dataset.

## Included pretrained assets

| File | Dataset | Latent dim | Pairs with config |
|---|---|---|---|
| `models/ae_phoenix_disentangled.pth` | PHOENIX-2014T | 80 (4×[8,28,28,16]) | `configs/train_phoenix.yaml` |
| `models/ae_phoenix_standard.pth` | PHOENIX-2014T | 80 (unified, ablation) | `configs/train_phoenix.yaml` |
| `models/ae_csl_disentangled.pth` | CSL-Daily | 80 (4×[8,28,28,16]) | `configs/train_CSL.yaml` |
| `data/channel_priors/channel_priors_phoenix_80dim.npy` | PHOENIX-2014T | 80 | Phase 2 `--prior_file` for the above |
| `data/channel_priors/channel_priors_csl_80dim.npy` | CSL-Daily | 80 | Phase 2 `--prior_file` for the above |
| `data/reference_pose.pt` | — | — | `DARSLPGenerator` decoder queries (default path, no flag needed) |

Only one config per dataset is needed — `configs/train_phoenix.yaml` and
`configs/train_CSL.yaml` don't encode latent dimensionality or model variant;
those are CLI flags (`--latent_dim`/`--face_latent_dim` on `train_stage1.py`,
`--pose_dim` on `train_stage2.py`, `--model standard` for the non-disentangled
ablation). `data/channel_priors/channel_priors_phoenix_{96,160}dim.npy` are
priors for the latent-dimension ablation in the paper — reused with
`configs/train_phoenix.yaml` at a different `--pose_dim`; no pretrained
checkpoint is shipped for those dimensions, train your own Stage 1 AE first.

## Usage — full pipeline (PHOENIX-2014T example)

```bash
# 0. Text embeddings (once per split)
python precompute_text_embeddings.py \
    --poses /data/phoenix/train.pt --tokenizer dbmdz/bert-base-german-uncased \
    --output_dir /data/phoenix/text_embeddings/train
# ... repeat for dev/test

# 1. Stage 1 — train the disentangled pose autoencoder
python train_stage1.py --model disentangled \
    --train_pt /data/phoenix/train.pt --train_embeddings /data/phoenix/text_embeddings/train \
    --dev_pt   /data/phoenix/dev.pt   --dev_embeddings   /data/phoenix/text_embeddings/dev \
    --output models/ae_phoenix_disentangled.pth --latent_dim 64 --face_latent_dim 16

# 2. Precompute Stage 1 latents (regression targets for Stage 2)
python precompute_encodings.py --model disentangled \
    --ae_ckpt models/ae_phoenix_disentangled.pth --poses /data/phoenix/train.pt \
    --output_dir data/pose_encodings/phoenix_train_disentangled_80dim
# ... repeat for dev

# 3. Stage 2, Phase 1 — latent regression + length prediction
python train_stage2.py --model darslp --config configs/train_phoenix.yaml \
    --ae_ckpt models/ae_phoenix_disentangled.pth \
    --train_pt /data/phoenix/train.pt --train_embeddings /data/phoenix/text_embeddings/train \
    --train_encodings data/pose_encodings/phoenix_train_disentangled_80dim \
    --dev_pt   /data/phoenix/dev.pt   --dev_embeddings   /data/phoenix/text_embeddings/dev \
    --dev_encodings   data/pose_encodings/phoenix_dev_disentangled_80dim \
    --pose_dim 80 --RH_weight 7 --LH_weight 5

# 4. Stage 2, Phase 2 — add channel-aware KL regularization
python train_stage2.py --model darslp --config configs/train_phoenix.yaml \
    --ae_ckpt models/ae_phoenix_disentangled.pth \
    --train_pt /data/phoenix/train.pt --train_embeddings /data/phoenix/text_embeddings/train \
    --train_encodings data/pose_encodings/phoenix_train_disentangled_80dim \
    --dev_pt   /data/phoenix/dev.pt   --dev_embeddings   /data/phoenix/text_embeddings/dev \
    --dev_encodings   data/pose_encodings/phoenix_dev_disentangled_80dim \
    --pose_dim 80 --RH_weight 7 --LH_weight 5 \
    --prior_file data/channel_priors/channel_priors_phoenix_80dim.npy

# 5. Inference
python src/prediction/predict_phoenix.py --config configs/train_phoenix.yaml \
    --ckpt /path/to/best-val-epoch=XXX.ckpt --ae_ckpt models/ae_phoenix_disentangled.pth \
    --text_embeddings /data/phoenix/text_embeddings/test --poses /data/phoenix/test.pt \
    --output predictions/predictions_phoenix_test.pt

# 6. Visualize predictions
python src/visualization/visualize_predictions_mixed.py \
    --predictions predictions/predictions_phoenix_test.pt \
    --output_dir visualizations/phoenix --num_samples 5

# 7. Export for back-translation evaluation
python src/prediction/export_phoenix.py --config configs/train_phoenix.yaml \
    --ckpt /path/to/best-val-epoch=XXX.ckpt --ae_ckpt models/ae_phoenix_disentangled.pth \
    --text_embeddings /data/phoenix/text_embeddings/test --poses /data/phoenix/test.pt \
    --output exports/predictions_DARSLP_phoenix_test.npz
```

CSL-Daily uses the equivalent `_csl.py` / `configs/train_CSL.yaml` counterparts
and `hfl/chinese-bert-wwm` as the tokenizer; see each script's own docstring
for the full argument list and dataset-specific notes (e.g. CSL's wrist-slot
convention in `predict_csl.py`).

To (re)compute channel priors for a new AE/dataset instead of using the ones
in `data/channel_priors/`, see `src/KL-div/compute_channel_priors.py` →
`save_channel_priors.py`.

## Evaluation

`export_*.py` output is consumed by external back-translation evaluation
pipelines (not part of this repo):
- PHOENIX-2014T: [SLRTP Sign Production Evaluation](https://github.com/walsharry/SLRTP-Sign-Production-Evaluation)
- CSL-Daily: [Sign Language Transformers (slt)](https://github.com/neccam/slt)

## Citation

```bibtex
@INPROCEEDINGS{11492489,
  author={Taşyürek, Sümeyye Meryem and Kızıltepe, Tuğçe and Keles, Hacer Yalim},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  title={Disentangle and Regularize: Sign Language Production with Articulator-Based Disentanglement and Channel-Aware Regularization},
  year={2026},
  volume={},
  number={},
  pages={8458-8467},
  keywords={Graphical user interfaces;Videos;Avatars;Protocols;Video equipment;HTTP;Wide area networks;Communication systems;Computer networks;Autoencoders;sign language production;text to pose generation;structured representation learning},
  doi={10.1109/WACV61042.2026.00816}}
```

## License

MIT — see [LICENSE](LICENSE).
