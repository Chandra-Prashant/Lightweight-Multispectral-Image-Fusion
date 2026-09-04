# Lightweight Multispectral Image Fusion

A small, decoupled two-stage network that fuses a noisy thermal/IR image
with a clean visible-light image, built and evaluated on the MSRS
(visible-infrared) dataset in `Thermal_Noise_Fusion_Experiment.ipynb`. The
core idea is separating *denoising* from *fusion* into two purpose-built
modules instead of one network trying to do both at once, so the fusion
step can explicitly reason about how much to trust the thermal input on a
per-pixel basis.

## Architecture

**Stage 1 - `LightweightThermalDenoiser`**: a small U-Net (2-level
encoder/decoder with skip connections, 16->32->64 channels) that denoises
the raw thermal channel. Trained with an MSE loss against clean ground
truth.

**Stage 2 - `NoiseAwareFusionLayer`**: takes the visible image, the
denoised thermal output, and a noise map (`|noisy_thermal -
denoised_thermal|`) as three inputs. It runs the noise map through a small
conv + sigmoid to get a per-pixel noise-confidence score, then computes an
**inverse attention mask** (`1.0 - noise_confidence`) - where the noise map
says a thermal pixel was heavily corrupted, that pixel's thermal features
get down-weighted before being concatenated with the visible features and
reconstructed into the fused output. This is the actual mechanism behind
what the author's profile describes as an "inverse-attention gating mask":
it lets the fusion stage lean on the visible image exactly where the
thermal sensor was least reliable, rather than trusting both modalities
uniformly.

Combined, the two modules total **117.5K parameters** (measured via
`thop.profile`, see Benchmarks below).

## Training

- **Dataset**: MSRS (Multi-Spectral Road Scenarios) visible/infrared pairs,
  1083 training images, loaded via a custom `MSRSFusionDataset` that also
  synthesizes the degradation the model is trained to handle: fixed-pattern
  vertical striping noise plus Gaussian sensor grain injected onto the
  thermal channel at load time.
- **Loss**: a weighted multi-task objective - denoiser MSE, fusion
  intensity L1 (retain thermal hot targets), fusion gradient L1 via Sobel
  filters weighted 2.0 (pull sharp structural edges from the visible
  image), and an SSIM structural loss weighted 5.0.
- **Setup**: Adam (lr 1e-3), batch size 4, 10 epochs on a Colab T4 GPU,
  checkpointed to disk every 5 epochs.

## Results (from the notebook's actual run, not estimated)

Measured on the held-out MSRS test split, with the same synthetic
degradation applied as at train time:

| Metric | Value |
|---|---|
| PSNR | 29.85 dB |
| SSIM | 0.9886 |
| Parameters | 117.52K |
| MACs | 1.836 G |
| FLOPs | 3.672 G |
| Inference latency | 4.21 ms/frame (T4 GPU) |
| Throughput | 237.4 FPS |

(If you've seen a PSNR of ~31 dB quoted elsewhere for this work, that's
from a different run than the one in this notebook - the checkpoint
evaluated here measured 29.85 dB. Worth re-running end to end and updating
either number to whichever is the one you want to stand behind.)

The notebook also runs a qualitative downstream check: it feeds the fused
output through a pretrained YOLOv5s detector to visually confirm object
detection still works on the fused image, not just that the pixel-level
metrics look good in isolation.

## Running

The notebook is written for Google Colab (mounts Google Drive for the
MSRS dataset and checkpoint storage) and needs a CUDA GPU for the timing
benchmark to be meaningful:

```bash
pip install torch torchvision timm scikit-image opencv-python thop matplotlib ultralytics
```

Update the hardcoded `/content/drive/MyDrive/Research_Project/...` dataset
and checkpoint paths near the top of the training/eval cells if running
outside Colab.

## Current state

This is a single-notebook research experiment, not a packaged library -
there's no `train.py`/`test.py`/CLI entry point, model code and training
loop live inline in the notebook cells, and dataset/checkpoint paths are
hardcoded to one person's Google Drive layout. The synthetic noise model
(fixed vertical striping + Gaussian grain) is also a simplification of
real thermal sensor degradation, not measured from real degraded sensor
data.
