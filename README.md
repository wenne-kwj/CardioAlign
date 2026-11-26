# CardioAlign

**Code repository accompanying the CardioAlign.**

CardioAlign is a paired brightfield–fluorescence imaging resource for human
induced pluripotent stem cell-derived cardiomyocyte cultures. Each field of view
contains a brightfield image and measured fluorescence images corresponding to
Hoechst, Calcein-AM and propidium iodide (PI) staining, linking cellular morphology
with nuclear localization, viability-associated signals and membrane
injury-associated signals.

This repository contains the model implementations, training and inference
code, evaluation configurations, and image-analysis tools accompanying the study.
It supports two tasks:

- **Virtual fluorescence prediction:** prediction of three-channel fluorescence
  from a single brightfield image, evaluated against measured fluorescence.
- **Quantitative image analysis:** nuclear and injury-associated object counting,
  foreground coverage and channel-specific signal quantification, with summaries
  across fields of view, wells, acquisition batches and experimental conditions.

## Models

Seven baselines cover supervised reconstruction, conditional adversarial
generation, diffusion and flow matching.

| Model | Architecture | Objective / sampling |
| --- | --- | --- |
| U-Net | Five-level convolutional encoder–decoder with GroupNorm and SiLU | Charbonnier + SSIM |
| Restormer | Four-level MDTA/GDFN backbone, width 32 | Charbonnier + SSIM |
| Pix2Pix-U-Net | Eight-downsampling U-Net generator and conditional PatchGAN | Adversarial BCE + L1 |
| Pix2Pix-ResNet | ResNet-50 encoder–decoder and adaptive conditional discriminator | Least-squares GAN + L1 |
| CellResDM | Swin-based conditional denoiser and residual diffusion | Clean-image MSE + LPIPS; residual reverse sampling |
| Palette | Conditional U-Net with continuous gamma conditioning and v prediction | v-MSE; DDIM |
| FlowDiT | BF-conditioned pixel DiT with a Gaussian-to-image flow | Clean-image parameterization of velocity loss; Euler / Heun |

Network definitions, model-specific objectives and sampling procedures are
organized under [models/](models/). Architecture, optimizer, loss and sampling
parameters are recorded in [configs/models/](configs/models/).

## Repository structure

```text
models/
├── unet/
├── restormer/
├── pix2pix_unet/
├── pix2pix_resnet/
├── cellresdm/
├── palette/
└── flowdit/
configs/
├── models/                  Model, loss, optimizer and sampling parameters
└── evaluation/              Image reconstruction and foreground metrics
engine/
├── registry.py              Model and objective construction
├── training.py              Training and weight updates
├── inference.py             Prediction and generative sampling
├── evaluation.py            Metric calculation and aggregation
├── image_metrics.py         Channel-wise image reconstruction metrics
└── scripts/
    ├── tiff_to_png.py       TIFF-to-PNG fluorescence rendering
    ├── quantify_png.py      Object counting and signal quantification
    ├── rgb_object_metrics.json
    ├── requirements.txt
    └── _lib/                Rendering, segmentation and object-matching code
scripts/
├── train.py
├── predict.py
└── evaluate.py
requirements/                Model and perceptual-loss dependencies
requirements.txt
LICENSE
```

## Virtual fluorescence prediction

The model interface uses floating-point tensors with one brightfield input
channel and three fluorescence output channels:

```text
Input:   [N, 1, H, W]
Target:  [N, 3, H, W]
```

External model inputs and targets use values in `[0, 1]`. GAN, diffusion and flow
models apply their signed-range mapping internally. Spatial dimensions follow
the selected backbone and model configuration. Channel order is defined by the
external image preparation and must remain consistent between targets and
predictions.

The entry points in [scripts/](scripts/) provide interfaces for training,
checkpoint-based prediction and image evaluation. Raw and exponential moving
average (EMA) weights can be selected for prediction when available.

Image evaluation includes MAE, relative-L2 NRMSE, SSIM, PCC and PSNR, with
per-sample and per-channel reporting. The foreground configuration adds pixel
Dice and IoU using independent Otsu thresholds. Metric settings are stored in
[configs/evaluation/](configs/evaluation/).

## Quantitative image analysis

The tools in [engine/scripts/](engine/scripts/) provide TIFF rendering and
quantitative analysis of fluorescence composites:

| Component | Function |
| --- | --- |
| [tiff_to_png.py](engine/scripts/tiff_to_png.py) | Render metadata-indexed fluorescence TIFFs as RGB16 PNG composites and RGB8 previews; record rendering windows, source hashes and export coverage. |
| [quantify_png.py](engine/scripts/quantify_png.py) | Compute watershed-separated red/blue object counts, red-to-blue count ratios, foreground coverage, mean intensity and integrated intensity. Export field-, well-, condition- and acquisition-level tables. |
| [rgb_object_metrics.json](engine/scripts/rgb_object_metrics.json) | Specify channel thresholds, minimum object areas and the instance-matching IoU threshold. |
| [_lib/counter_cli.py](engine/scripts/_lib/counter_cli.py) | Count objects in individual images or compare paired predictions and references, including count error, foreground overlap and instance precision, recall and F1. |

The image-analysis scripts take an external resource root through
`--dataset-root` and use its image metadata to locate corresponding files.
Input and output locations are supplied at runtime.

TIFF rendering applies recorded per-field display windows. Intensity statistics
from the resulting PNGs are therefore expressed on the rendered PNG scale;
acquisition-scale fluorescence intensity analysis uses the original TIFF values.
Automated object counts and foreground masks are image-derived measurements,
interpreted with the corresponding staining and acquisition metadata.

## Dependencies

Model code uses Python 3.10+ and PyTorch 2.2+. Dependency files are grouped by
function:

```bash
# Model dependencies
pip install -r requirements.txt

# TIFF rendering, segmentation and quantitative image analysis
pip install -r engine/scripts/requirements.txt

# LPIPS perceptual loss for CellResDM
pip install -r requirements/perceptual.txt
```

CellResDM's perceptual loss uses an externally prepared pretrained-weight cache,
specified by `loss.pretrained_cache`. Its contribution is controlled by
`loss.lpips_weight`.

## Citation and attribution

Please cite the CardioAlign study and the relevant model publications when using
this repository. Upstream attributions are retained in the model source files.
Restormer's MIT notice is provided in
[LICENSE.Restormer](models/restormer/LICENSE.Restormer), and the original pix2pix
notices are provided in [LICENSE.pix2pix](models/pix2pix_unet/LICENSE.pix2pix).

See [LICENSE](LICENSE) for repository-level licensing information.
