# FakeDetect AI — AI-Generated Image Detector

A deep learning web application that detects whether an image is real or AI-generated. Built with PyTorch (MobileNetV2), exported to ONNX, and deployed as a static frontend using ONNX Runtime Web.

**Live Demo:** https://fake-image-detector-three.vercel.app

---

## Project Overview

| Property | Details |
|---|---|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | CIFAKE — Real and AI-Generated Synthetic Images |
| Training images | 4,000 (2,000 real, 2,000 fake) |
| Validation accuracy | 94% |
| Epochs | 15 |
| Deployment | Vercel (static hosting) |
| In-browser inference | ONNX Runtime Web |

---

## Requirements

### Dataset

- **CIFAKE** — Real and AI-Generated Synthetic Images
  - Source: [Kaggle — birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  - Size: ~300MB (zip)
  - Contents: 140,000 images split into `REAL` and `FAKE` classes, pre-divided into `train/` and `test/` folders
  - You only need ~2,000 images per class for training

### System Requirements

| Requirement | Details |
|---|---|
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.10 or higher |
| GPU | NVIDIA GPU recommended (tested on RTX 4060) |
| CUDA | 12.x (driver version 520+) |
| RAM | 8GB minimum, 16GB recommended |
| Storage | ~2GB free space |

### Python Libraries

Install all dependencies with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install onnx onnxruntime
pip install Pillow matplotlib scikit-learn
```

> If you do not have an NVIDIA GPU, omit the `--index-url` flag to install the CPU-only version of PyTorch. Training will be significantly slower.

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
├── style.css               # Stylesheet
├── app.js                  # Frontend logic + ONNX inference
├── model.onnx              # Exported model (browser-ready)
├── README.md               # This file
└── model/
    ├── train.py            # Model training script
    ├── export.py           # ONNX export script
    ├── model.pth           # Saved PyTorch weights
    ├── training_plot.png   # Accuracy graph (generated after training)
    └── my_dataset/
        ├── REAL/           # 2,000 real images
        └── FAKE/           # 2,000 fake images
```

---

## Step-by-Step Build Guide

### Phase 0 — Prepare the Dataset

**1. Download CIFAKE from Kaggle**

Go to the dataset page, click Download, and extract the zip file. You will see:

```
cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

**2. Create a subset for training**

Teachable Machine and local training work best with 1,000–2,000 images per class. Create a script called `prepare_data.py` inside the `model/` folder:

```python
import os, shutil, random

SOURCE_REAL = "cifake/train/REAL"
SOURCE_FAKE = "cifake/train/FAKE"
DEST_REAL   = "my_dataset/REAL"
DEST_FAKE   = "my_dataset/FAKE"
NUM_IMAGES  = 2000

os.makedirs(DEST_REAL, exist_ok=True)
os.makedirs(DEST_FAKE, exist_ok=True)

for src, dst in [(SOURCE_REAL, DEST_REAL), (SOURCE_FAKE, DEST_FAKE)]:
    images   = os.listdir(src)
    selected = random.sample(images, NUM_IMAGES)
    for img in selected:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("Done! Dataset prepared.")
```

Run it:

```bash
cd model
python prepare_data.py
```

---

### Phase 1 — Build the Frontend

Create three files in the root of the project: `index.html`, `style.css`, and `app.js`.

At this stage the frontend is a static shell — upload area, image preview, result display — with a placeholder result. The real AI prediction is wired in during Phase 3.

The frontend uses:
- Vanilla HTML, CSS, and JavaScript — no frameworks
- ONNX Runtime Web (loaded from CDN) for in-browser inference
- Drag-and-drop and file picker upload support

---

### Phase 2 — Train the Model

**1. Verify your GPU is detected**

Create `check_gpu.py`:

```python
import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

```bash
python check_gpu.py
```

**2. Train using MobileNetV2 with Transfer Learning**

Run the training script from inside the `model/` folder:

```bash
cd model
python train.py
```

The script:
- Loads your `my_dataset/` folder using `torchvision.datasets.ImageFolder`
- Splits 80% train / 20% validation automatically
- Loads pretrained MobileNetV2 weights from ImageNet
- Freezes all layers except the last 3 blocks and the classifier
- Trains for 15 epochs with a StepLR learning rate scheduler
- Saves the best model (by validation accuracy) to `model.pth`
- Generates a `training_plot.png` accuracy graph

Expected training time: 8–12 minutes on an NVIDIA RTX 4060.

Expected output:
```
Training on: cuda
Classes: ['FAKE', 'REAL']
Total images: 4000
Epoch 1/15  — Train: 72.3%  Val: 75.1%
...
Epoch 15/15 — Train: 95.2%  Val: 94.0%  ✓ Best model saved
Training complete. Best val accuracy: 94.0%
```

**3. Export to ONNX**

PyTorch models cannot run directly in the browser. Export to ONNX format:

```bash
python export.py
```

This creates `model.onnx` (~8.5 MB), which is loaded by ONNX Runtime Web in the browser.

---

### Phase 3 — Integrate the Model

Copy `model.onnx` from `model/` into the root of the project.

The `app.js` file handles the full inference pipeline:

1. Loads `model.onnx` using `ort.InferenceSession.create()`
2. When an image is uploaded, resizes it to 224×224 on a canvas
3. Applies the same ImageNet normalisation used during training:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
4. Converts pixel data to a `Float32Array` in CHW format (channels first)
5. Runs inference via `session.run()`
6. Applies softmax to raw logits to get probabilities
7. Displays the predicted class, confidence percentage, and both class probabilities

**Test locally:**

```bash
python -m http.server 8080
```

Open `http://localhost:8080` in your browser. Opening `index.html` directly as a file will not work because the browser blocks local file loading of `.onnx` files for security reasons.

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

> For the GitHub password prompt, use a Personal Access Token (not your account password). Generate one at: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → tick `repo`.

**2. Deploy on Vercel**

1. Go to vercel.com and sign in with GitHub
2. Click **Add New Project**
3. Select your `fake-image-detector` repository
4. Leave all settings as default — Vercel auto-detects it as a static site
5. Click **Deploy**

Your app will be live at a URL like:
```
https://fake-image-detector-xyz.vercel.app
```

---

## How It Works

```
User uploads image
       ↓
Canvas resizes to 224×224
       ↓
Pixel values normalised (ImageNet mean/std)
       ↓
Float32Array converted to CHW tensor
       ↓
ONNX Runtime runs MobileNetV2 inference
       ↓
Softmax converts logits to probabilities
       ↓
Result displayed (FAKE / REAL + confidence %)
```

---

## Model Architecture

MobileNetV2 is a lightweight convolutional neural network designed for mobile and edge devices. It uses depthwise separable convolutions and inverted residual blocks to achieve high accuracy with low compute cost.

For this project:
- All layers were initially frozen (pretrained ImageNet weights preserved)
- The last 3 convolutional blocks were unfrozen for fine-tuning
- The final classifier layer was replaced with a 2-class output (REAL / FAKE)
- Only the unfrozen layers and new classifier were trained

This technique — **Transfer Learning** — allows high accuracy with far fewer training images than training from scratch would require.

---

## Limitations

- The model is optimised specifically for **Stable Diffusion** generated images (the AI generator used in the CIFAKE dataset)
- Images generated by other AI tools such as Midjourney, DALL-E, or Flux may not be classified correctly, as their artifact patterns differ
- Heavily compressed images (from social media platforms like Instagram or WhatsApp) may produce unreliable results due to compression altering pixel-level patterns
- The model was trained on 32×32 images upscaled to 224×224 — very high resolution real-world photos may behave differently

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch 2.x, torchvision |
| Base model | MobileNetV2 (pretrained on ImageNet) |
| Dataset | CIFAKE (Krizhevsky et al.) |
| Model export | ONNX (opset 18) |
| In-browser inference | ONNX Runtime Web |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Hosting | Vercel (free tier) |

---

## Acknowledgements

- CIFAKE dataset: Bird & Lotfi, 2023 — [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- MobileNetV2: Sandler et al., 2018 — [arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- ONNX Runtime Web: [onnxruntime.ai](https://onnxruntime.ai)

---

*Built as part of the Applied AI Mini Project — First Year, 2026.*
