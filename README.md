# VisionScan AI — Real or Fake Image Detector

A deep learning web application that detects whether an image is real or AI-generated. Built with PyTorch (MobileNetV2), exported to ONNX, and deployed as a fully static frontend using ONNX Runtime Web — no server, no API key, no data leaves your device.

**Live Demo:** https://fake-image-detector-three.vercel.app

---

## Project Overview

| Property | Details |
|---|---|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | CIFAKE + Tristan Zhang Dataset |
| Training images | 4,000 (2,000 real, 2,000 fake) |
| Validation accuracy | 93% |
| Epochs | 15 |
| Deployment | Vercel (static hosting) |
| In-browser inference | ONNX Runtime Web |

---

## Features

- Upload via drag & drop, file picker, or Ctrl+V paste from clipboard
- Analysed image stays visible alongside the verdict — no disappearing preview
- Shows both real and fake probability scores side by side
- Confidence bar with animated fill
- Fully client-side — model runs in the browser, no data sent to any server
- Supports JPG, PNG, WEBP up to 50MB
- Responsive layout — works on mobile and desktop

---

## Requirements

### Datasets

**Dataset 1 — CIFAKE**
- Source: [Kaggle — birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Size: ~300MB (zip)
- Contents: 140,000 images split into `REAL` and `FAKE` classes, pre-divided into `train/` and `test/` folders
- Used for: FAKE class (Stable Diffusion generated images)

**Dataset 2 — Tristan Zhang AI vs Real Images**
- Source: [Kaggle — tristanzhang32/ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
- Size: ~30,000 images
- Structure: `train/real/` and `train/fake/`
- Used for: REAL class (diverse web-scraped real photographs) + additional FAKE diversity

> Using both datasets together gives the model exposure to diverse real-world photographs and multiple AI generators, improving generalisation beyond just Stable Diffusion images.

### System Requirements

| Requirement | Details |
|---|---|
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.10 or higher |
| GPU | NVIDIA GPU recommended (tested on RTX 4060) |
| CUDA | 12.x or higher (driver version 520+) |
| RAM | 8GB minimum, 16GB recommended |
| Storage | ~3GB free space |

### Python Libraries

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install onnx onnxruntime
pip install Pillow matplotlib scikit-learn
```

> If you do not have an NVIDIA GPU, omit the `--index-url` flag to install the CPU-only version of PyTorch. Training will be significantly slower.

> `cu128` targets CUDA 12.8, which is compatible with CUDA 13.x drivers due to NVIDIA's backwards compatibility guarantee.

### Tools

| Tool | Purpose | Download |
|---|---|---|
| Python 3.10+ | Training and export scripts | python.org |
| VS Code | Code editor | code.visualstudio.com |
| Git | Version control and deployment | git-scm.com |
| Kaggle account | Dataset download | kaggle.com |
| GitHub account | Repository hosting | github.com |
| Vercel account | Free deployment | vercel.com |

---

## Project Structure

```
fake-image-detector/
├── index.html              # Main frontend page
├── style.css               # All styles
├── app.js                  # Frontend logic + ONNX inference
├── model.onnx              # Exported model (browser-ready, ~8.5MB)
├── README.md               # This file
└── model/
    ├── train.py            # Model training script
    ├── export.py           # ONNX export script
    ├── check_gpu.py        # GPU verification script
    ├── prepare_data.py     # Dataset preparation script
    ├── model.pth           # Saved PyTorch weights (best epoch)
    ├── training_plot.png   # Accuracy graph (generated after training)
    └── my_dataset/
        ├── REAL/           # 2,000 real images (Tristan Zhang only)
        └── FAKE/           # 2,000 fake images (1,000 CIFAKE + 1,000 Tristan Zhang)
```

---

## Step-by-Step Build Guide

### Phase 0 — Prepare the Dataset

**1. Download both datasets from Kaggle**

Extract both zip files. After extracting you should have:

```
cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/

tristanzhang/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**2. Prepare the combined dataset**

Create `prepare_data.py` inside the `model/` folder:

```python
import os, shutil, random

# Real images — Tristan Zhang only
# CIFAKE real images are 32x32 upscaled CIFAR images — too low quality for real-world generalisation
TRISTAN_REAL = "tristanzhang/train/real"

# Fake images — both sources for generator diversity
CIFAKE_FAKE  = "cifake/train/FAKE"
TRISTAN_FAKE = "tristanzhang/train/fake"

DEST_REAL = "my_dataset/REAL"
DEST_FAKE = "my_dataset/FAKE"

os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

# Clear existing dataset
for f in os.listdir(DEST_REAL): os.remove(os.path.join(DEST_REAL, f))
for f in os.listdir(DEST_FAKE): os.remove(os.path.join(DEST_FAKE, f))

def copy_sample(src, dst, n, prefix):
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    if len(images) < n:
        print(f"WARNING: Only {len(images)} found in {src}, using all")
        n = len(images)
    selected = random.sample(images, n)
    for i, img in enumerate(selected):
        ext = os.path.splitext(img)[1]
        shutil.copy(os.path.join(src, img), os.path.join(dst, f"{prefix}_{i}{ext}"))
    print(f"Copied {len(selected)} from {src}")

copy_sample(TRISTAN_REAL, DEST_REAL, 2000, "real")
copy_sample(CIFAKE_FAKE,  DEST_FAKE, 1000, "cifake_fake")
copy_sample(TRISTAN_FAKE, DEST_FAKE, 1000, "tristan_fake")

print(f"\nDone!")
print(f"REAL: {len(os.listdir(DEST_REAL))} images")
print(f"FAKE: {len(os.listdir(DEST_FAKE))} images")
```

```bash
cd model
python prepare_data.py
```

---

### Phase 1 — Build the Frontend

The frontend consists of three files: `index.html`, `style.css`, and `app.js`.

Key frontend features:
- Animated gradient border upload card with drag-and-drop
- Clipboard paste support — right-click any web image → Copy Image → Ctrl+V
- After analysis: two-column layout with the analysed image on the left and verdict on the right
- The uploaded image stays visible alongside the result
- Animated confidence bar and dual probability display
- Animated grid background, floating orbs, glitch title effect

No frameworks — pure HTML, CSS, and JavaScript.

---

### Phase 2 — Train the Model

**1. Verify your GPU**

```python
# check_gpu.py
import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

```bash
python check_gpu.py
```

**2. Train**

```bash
cd model
python train.py
```

The training script:
- Loads `my_dataset/` using `torchvision.datasets.ImageFolder`
- Splits 80% train / 20% validation automatically
- Loads pretrained MobileNetV2 weights from ImageNet
- Freezes all base layers, unfreezes the last 3 convolutional blocks for fine-tuning
- Replaces the final classifier with a 2-class output (REAL / FAKE)
- Applies data augmentation — random flips, rotation, colour jitter, grayscale
- Trains for 15 epochs with StepLR scheduler (step=3, gamma=0.5)
- Saves the best checkpoint to `model.pth`
- Generates `training_plot.png`

Expected training time: 8–12 minutes on an NVIDIA RTX 4060.

Expected output:
```
Training on: cuda
Classes: ['FAKE', 'REAL']
Total images: 4000
Epoch 1/15  — Train: 68.4%  Val: 71.2%
...
Epoch 15/15 — Train: 94.8%  Val: 93.0%  ✓ Best model saved
Training complete. Best val accuracy: 93.0%
```

> **Windows users:** If you get a multiprocessing bootstrap error, ensure training code is inside `if __name__ == '__main__':` and set `num_workers=0` in both DataLoader calls.

**3. Export to ONNX**

```bash
python export.py
```

Creates `model.onnx` (~8.5MB) using opset 18 with `dynamo=False`.

---

### Phase 3 — Integrate the Model

Copy `model.onnx` from `model/` into the project root.

The `app.js` inference pipeline:

1. Loads `model.onnx` on page load via `ort.InferenceSession.create()`
2. Accepts images via drag & drop, file picker, or Ctrl+V paste
3. Resizes image to 224×224 on an HTML canvas
4. Applies ImageNet normalisation matching training:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
5. Converts to `Float32Array` in CHW format
6. Runs `session.run({ input: tensor })`
7. Applies softmax to get probabilities
8. Displays image + verdict side by side with confidence and both class probabilities

**Test locally:**

```bash
python -m http.server 8080
```

Open `http://localhost:8080`. Do not open `index.html` directly — browsers block loading `.onnx` files from `file://` URLs.

---

### Phase 4 — Deploy on Vercel

**1. Push to GitHub**

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fake-image-detector.git
git push -u origin main
```

> Use a Personal Access Token for the password — GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → tick `repo`.

**2. Deploy**

1. Go to vercel.com → sign in with GitHub
2. Click **Add New Project** → select your repository
3. Leave all settings as default
4. Click **Deploy**

Live in ~30 seconds. Subsequent pushes to GitHub trigger automatic redeployment.

---

## How It Works

```
User uploads image (drag & drop / file picker / Ctrl+V paste)
       ↓
HTML canvas resizes image to 224×224
       ↓
Pixel values normalised per channel (ImageNet mean & std)
       ↓
Float32Array reshaped to [1, 3, 224, 224] CHW tensor
       ↓
ONNX Runtime Web runs MobileNetV2 inference in browser
       ↓
Softmax converts raw logits → probabilities
       ↓
Image + verdict displayed side by side (REAL / FAKE + confidence %)
```

---

## Model Architecture

MobileNetV2 uses depthwise separable convolutions and inverted residual blocks — high accuracy with low parameter count, ideal for browser deployment.

**Transfer Learning approach:**
- Base model pretrained on ImageNet (1.2M images, 1,000 classes)
- All layers frozen — preserves general visual feature extraction
- Last 3 convolutional blocks unfrozen — fine-tuned for real vs fake distinction
- Final classifier replaced with 2-class linear layer
- Only unfrozen layers + classifier trained on 4,000 images

This achieves 93% accuracy with a small dataset — training from scratch would require far more data.

---

## Dataset Strategy

| Class | Source | Reason |
|---|---|---|
| REAL | Tristan Zhang only | Diverse web-scraped photos — generalises to real-world images |
| FAKE | CIFAKE (1,000) + Tristan Zhang (1,000) | Covers Stable Diffusion and other generators |

> CIFAKE real images were excluded from the REAL class intentionally. They are upscaled 32×32 CIFAR images — blurry and unnatural. Including them caused the model to associate blurriness with "real", hurting performance on actual photographs.

---

## Limitations

- Optimised primarily for Stable Diffusion generated images
- Other generators (Midjourney, Firefly, Flux) may produce less reliable results
- Heavily compressed images (WhatsApp, Instagram) alter pixel patterns and can confuse the model
- The model classifies the full image — it cannot localise which region appears AI-generated

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch 2.x, torchvision |
| Base model | MobileNetV2 (pretrained on ImageNet) |
| Dataset 1 | CIFAKE — Bird & Lotfi, 2023 |
| Dataset 2 | Tristan Zhang AI vs Real Images |
| Model export | ONNX (opset 18) |
| In-browser inference | ONNX Runtime Web (CDN) |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Fonts | Orbitron, Syne (Google Fonts) |
| Hosting | Vercel (free tier) |

---

## Acknowledgements

- CIFAKE dataset: Bird & Lotfi, 2023 — [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Tristan Zhang dataset — [Kaggle](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
- MobileNetV2: Sandler et al., 2018 — [arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- ONNX Runtime Web — [onnxruntime.ai](https://onnxruntime.ai)

---

*Built as part of the Applied AI Mini Project — CSE-Cyber (Hons.), Semester 2 (2025-26)*
*Karnavati University — UnitedWorld Institute of Technology*