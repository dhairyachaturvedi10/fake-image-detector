const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const previewImg    = document.getElementById('previewImg');
const imgFilename   = document.getElementById('imgFilename');
const loadingState  = document.getElementById('loadingState');
const resultState   = document.getElementById('resultState');
const verdictChip   = document.getElementById('verdictChip');
const verdictIcon   = document.getElementById('verdictIcon');
const verdictLabel  = document.getElementById('verdictLabel');
const verdictSub    = document.getElementById('verdictSub');
const confPct       = document.getElementById('confPct');
const barFill       = document.getElementById('barFill');
const fakeProb      = document.getElementById('fakeProb');
const realProb      = document.getElementById('realProb');
const resetBtn      = document.getElementById('resetBtn');

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

function preprocessImage(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width  = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgElement, 0, 0, 224, 224);

  const imageData = ctx.getImageData(0, 0, 224, 224);
  const { data }  = imageData;

  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const float32 = new Float32Array(1 * 3 * 224 * 224);

  for (let i = 0; i < 224 * 224; i++) {
    const r = data[i * 4]     / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float32[0 * 224 * 224 + i] = (r - mean[0]) / std[0];
    float32[1 * 224 * 224 + i] = (g - mean[1]) / std[1];
    float32[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', float32, [1, 3, 224, 224]);
}

function softmax(arr) {
  const max  = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

async function predict(imgElement) {
  if (!session) {
    alert('Model is still loading, please try again in a moment.');
    return null;
  }
  const tensor = preprocessImage(imgElement);
  const output = await session.run({ input: tensor });
  const probs  = softmax(Array.from(output.output.data));

  const fakeScore = Math.round(probs[0] * 100);
  const realScore = Math.round(probs[1] * 100);
  const isFake    = fakeScore >= realScore;

  return {
    label:      isFake ? 'FAKE' : 'REAL',
    confidence: isFake ? fakeScore : realScore,
    fakeScore,
    realScore,
  };
}

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleImage(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleImage(fileInput.files[0]);
});

function handleImage(file) {
  const reader = new FileReader();
  imgFilename.textContent = file.name;

  reader.onload = async (e) => {
    previewImg.src = e.target.result;

    uploadArea.style.display    = 'none';
    resultSection.style.display = 'grid';
    loadingState.style.display  = 'flex';
    resultState.style.display   = 'none';

    previewImg.onload = async () => {
      const result = await predict(previewImg);
      if (result) showResult(result);
    };
  };
  reader.readAsDataURL(file);
}

function showResult(result) {
  loadingState.style.display = 'none';
  resultState.style.display  = 'flex';

  const isFake = result.label === 'FAKE';

  verdictChip.className  = 'verdict-chip ' + (isFake ? 'fake' : 'real');
  verdictIcon.className  = 'vicon ' + (isFake ? 'fake' : 'real');
  verdictIcon.textContent = isFake ? '✗' : '✓';
  verdictLabel.className  = 'vlabel ' + (isFake ? 'fake' : 'real');
  verdictLabel.textContent = isFake ? 'AI Generated' : 'Real Photo';
  verdictSub.textContent  = result.confidence >= 80
    ? 'Detected with high confidence'
    : 'Detected with moderate confidence';

  confPct.textContent      = result.confidence + '%';
  barFill.className        = 'bar-fill ' + (isFake ? 'fake' : 'real');
  barFill.style.width      = result.confidence + '%';

  fakeProb.className       = 'prob-val fake';
  fakeProb.textContent     = result.fakeScore + '%';
  realProb.className       = 'prob-val real';
  realProb.textContent     = result.realScore + '%';
}

resetBtn.addEventListener('click', () => {
  uploadArea.style.display    = 'block';
  resultSection.style.display = 'none';
  fileInput.value             = '';
  barFill.style.width         = '0%';
});