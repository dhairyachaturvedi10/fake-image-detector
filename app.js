const CLASS_NAMES = ['FAKE', 'REAL'];
const MODEL_PATH  = 'model.onnx';
let session = null;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create(MODEL_PATH);
    console.log('Model loaded successfully');
  } catch (e) {
    console.error('Failed to load model:', e);
  }
}
loadModel();

// ── Preprocessing (matches val_transforms in train.py) ──
function preprocessImage(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgElement, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);

  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const float32 = new Float32Array(3 * 224 * 224);

  for (let i = 0; i < 224 * 224; i++) {
    float32[0 * 224 * 224 + i] = (data[i*4]   / 255 - mean[0]) / std[0];
    float32[1 * 224 * 224 + i] = (data[i*4+1] / 255 - mean[1]) / std[1];
    float32[2 * 224 * 224 + i] = (data[i*4+2] / 255 - mean[2]) / std[2];
  }
  return new ort.Tensor('float32', float32, [1, 3, 224, 224]);
}

function softmax(arr) {
  const max  = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// ── Upload handlers ───────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Clipboard paste (Ctrl+V) ──────────────────────────
document.addEventListener('paste', e => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) handleFile(file);
      break;
    }
  }
});

// ── Main file handler ─────────────────────────────────
function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png|webp)/)) {
    alert('Please upload a JPG, PNG, or WEBP image.');
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    alert('File size must be under 50MB.');
    return;
  }

  const reader = new FileReader();
  reader.onload = e => {
    const src = e.target.result;

    // Set image on result panel immediately
    const resultImg = document.getElementById('result-img');
    resultImg.src = src;

    // Hide upload card, show loading inside it
    document.getElementById('upload-prompt').style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';

    // Wait for image to render, then run inference
    resultImg.onload = () => {
      setTimeout(() => analyzeImage(resultImg), 600);
    };

    // If image is already cached (same src), onload won't fire — trigger manually
    if (resultImg.complete) {
      setTimeout(() => analyzeImage(resultImg), 600);
    }
  };
  reader.readAsDataURL(file);
}

// ── ONNX inference ────────────────────────────────────
async function analyzeImage(imgElement) {
  if (!session) {
    showResult({ verdict: 'REAL', confidence: 0, fakeScore: 0, realScore: 0, reason: 'Model still loading — please try again in a moment.' });
    return;
  }

  try {
    const tensor = preprocessImage(imgElement);
    const output = await session.run({ input: tensor });
    const probs  = softmax(Array.from(output.output.data));

    const fakeScore = Math.round(probs[0] * 100);
    const realScore = Math.round(probs[1] * 100);
    const isFake    = fakeScore >= realScore;

    showResult({
      verdict:    isFake ? 'AI-GENERATED' : 'REAL',
      confidence: isFake ? fakeScore : realScore,
      fakeScore,
      realScore,
      reason: isFake
        ? `AI-generated patterns detected — ${fakeScore}% fake / ${realScore}% real`
        : `Real photograph characteristics identified — ${realScore}% real / ${fakeScore}% fake`
    });
  } catch (err) {
    console.error(err);
    showResult({ verdict: 'REAL', confidence: 0, fakeScore: 0, realScore: 0, reason: 'Analysis failed — please try again.' });
  }
}

// ── Show result ───────────────────────────────────────
function showResult({ verdict, confidence, fakeScore, realScore, reason }) {
  // Hide upload card entirely
  document.getElementById('upload-card').style.display = 'none';

  // Show result layout (image + verdict side by side)
  const resultLayout = document.getElementById('result-layout');
  resultLayout.style.display = 'grid';

  // Show result section
  const resultSection = document.getElementById('result-section');
  resultSection.style.display = 'block';

  const isReal = verdict === 'REAL';

  // Verdict card
  const card      = document.getElementById('result-card');
  const verdictEl = document.getElementById('result-verdict');
  const subEl     = document.getElementById('result-sub');
  const confPct   = document.getElementById('conf-pct');
  const confBar   = document.getElementById('conf-bar');
  const realProb  = document.getElementById('real-prob');
  const fakeProb  = document.getElementById('fake-prob');

  card.className        = 'result-card ' + (isReal ? 'real' : 'fake');
  verdictEl.textContent = isReal ? '✓ Real Photo' : '⚡ AI Generated';
  subEl.textContent     = reason;
  confPct.textContent   = confidence + '%';

  // Probability values with color classes
  realProb.className    = 'prob-val real-val';
  realProb.textContent  = realScore + '%';
  fakeProb.className    = 'prob-val fake-val';
  fakeProb.textContent  = fakeScore + '%';

  // Image badge
  document.getElementById('img-badge').textContent = 'Analysed Image';

  // Animate confidence bar
  requestAnimationFrame(() => requestAnimationFrame(() => {
    confBar.style.width = confidence + '%';
  }));
}

// ── Reset ─────────────────────────────────────────────
function resetAll() {
  // Show upload card again
  document.getElementById('upload-card').style.display = 'block';
  document.getElementById('upload-prompt').style.display = 'block';
  document.getElementById('loading-section').style.display = 'none';

  // Hide result layout
  document.getElementById('result-layout').style.display = 'none';
  document.getElementById('result-section').style.display = 'none';

  // Reset bar and input
  document.getElementById('conf-bar').style.width = '0%';
  document.getElementById('result-img').src = '';
  fileInput.value = '';
}