const { readJson, refreshSummary, escapeHtml } = window.CatIdentity;

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const resultState = document.getElementById('resultState');
const resultCard = document.getElementById('resultCard');
const finalLabel = document.getElementById('finalLabel');
const scorePill = document.getElementById('scorePill');
const analysisPanel = document.getElementById('analysisPanel');
const analysisFrame = document.getElementById('analysisFrame');
const analysisPreview = document.getElementById('analysisPreview');
const analysisSummary = document.getElementById('analysisSummary');
const analysisCropBox = document.getElementById('analysisCropBox');
const analysisFaceBox = document.getElementById('analysisFaceBox');
const analysisSteps = document.getElementById('analysisSteps');
const analysisNote = document.getElementById('analysisNote');
const modelInputPanel = document.getElementById('modelInputPanel');
const modelInputPreview = document.getElementById('modelInputPreview');
const modelInputMeta = document.getElementById('modelInputMeta');
const modelInputNote = document.getElementById('modelInputNote');
const profileDetails = document.getElementById('profileDetails');
const resultSimilarity = document.getElementById('resultSimilarity');
const resultCatName = document.getElementById('resultCatName');
const resultOwner = document.getElementById('resultOwner');
const resultContact = document.getElementById('resultContact');
const resultLocation = document.getElementById('resultLocation');
const statusDetails = document.getElementById('statusDetails');
const statusTitle = document.getElementById('statusTitle');
const statusCopy = document.getElementById('statusCopy');
const bestKnown = document.getElementById('bestKnown');
const secondScore = document.getElementById('secondScore');
const bestUnknown = document.getElementById('bestUnknown');
const bestNotCat = document.getElementById('bestNotCat');
const qualityNotes = document.getElementById('qualityNotes');
const majorityVotePanel = document.getElementById('majorityVotePanel');
const decisionSummaryText = document.getElementById('decisionSummaryText');
const winnerLabel = document.getElementById('winnerLabel');
const winnerVotes = document.getElementById('winnerVotes');
const winnerTop10Votes = document.getElementById('winnerTop10Votes');
const winnerWeightedSum = document.getElementById('winnerWeightedSum');
const runnerUpLabel = document.getElementById('runnerUpLabel');
const decisionReason = document.getElementById('decisionReason');
const topCandidates = document.getElementById('topCandidates');
const classSummaryList = document.getElementById('classSummaryList');
const resultFootnote = document.getElementById('resultFootnote');
const resultActions = document.getElementById('resultActions');
const viewProfileBtn = document.getElementById('viewProfileBtn');
const retryPredictBtn = document.getElementById('retryPredictBtn');
const video = document.getElementById('video');
const cameraCanvas = document.getElementById('cameraCanvas');
const startCameraBtn = document.getElementById('startCameraBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const captureBtn = document.getElementById('captureBtn');

const state = { currentFile: null, capturedDataUrl: null, stream: null, cameraFacingMode: 'environment', analysisTimers: [] };
const labelMap = { unknown_cat: 'แมวที่ไม่อยู่ในระบบ', not_cat: 'ไม่ใช่แมว', low_quality: 'คุณภาพภาพไม่ผ่าน' };
const qualityReasonMap = { image_too_small: 'รูปมีขนาดเล็กเกินไป', image_too_blurry: 'รูปเบลอเกินไป', image_too_dark: 'รูปมืดเกินไป', image_too_bright: 'รูปสว่างเกินไป' };
const cropStrategyMap = { cat_face: 'ครอปตามหน้าแมว', center_square: 'ครอปกลางภาพ' };
const faceDetectionSourceMap = { 'yolo+face_model': 'YOLO + face model', yolo: 'YOLO detector', 'haar+face_model': 'Haar + face model', haar: 'Haar', face_model: 'face model', none: 'ไม่พบหน้าแมว' };

function formatLabel(label) { return labelMap[label] || label || '-'; }
function formatScore(value, digits = 4) { const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits) : '-'; }
function formatText(value, fallback = 'ยังไม่ระบุ') { const t = String(value ?? '').trim(); return t || fallback; }
function formatQualityReasons(reasons) { const labels = (reasons || []).map((reason) => qualityReasonMap[reason] || reason); return labels.length ? labels.join(', ') : 'ไม่มี'; }
function formatCropStrategy(value) { return cropStrategyMap[value] || formatText(value, '-'); }
function formatFaceDetectionSource(value) { return faceDetectionSourceMap[value] || formatText(value, '-'); }
function uploadedFileUrl(filePath) {
  const clean = String(filePath ?? '').trim().replace(/^\/+/, '');
  if (!clean) return '';
  return `/uploads/${clean.split('/').map((part) => encodeURIComponent(part)).join('/')}`;
}
function resetAnalysisPlayback() {
  state.analysisTimers.forEach((timer) => window.clearTimeout(timer));
  state.analysisTimers = [];
  if (!analysisFrame || !analysisSteps) return;
  analysisFrame.classList.remove('play-scan');
  analysisSteps.querySelectorAll('li').forEach((item) => item.classList.remove('active'));
}
function scheduleAnalysisStep(delay, callback) {
  const timer = window.setTimeout(callback, delay);
  state.analysisTimers.push(timer);
}
function setAbsoluteBoxStyle(element, box, imageWidth, imageHeight) {
  if (!box || !imageWidth || !imageHeight) {
    element.classList.add('hidden');
    element.removeAttribute('style');
    return;
  }
  const [x, y, width, height] = box.map((value) => Number(value) || 0);
  element.style.left = `${(x / imageWidth) * 100}%`;
  element.style.top = `${(y / imageHeight) * 100}%`;
  element.style.width = `${(width / imageWidth) * 100}%`;
  element.style.height = `${(height / imageHeight) * 100}%`;
  element.classList.remove('hidden');
}

function statusSummary(data) {
  if (data.final_label === 'not_cat') return { title: 'ไม่ใช่แมว', copy: data.decision_reason || 'Top 10 เอนเอียงไปที่ not_cat มากที่สุด' };
  if (data.final_label === 'low_quality') return { title: 'คุณภาพภาพไม่ผ่าน', copy: data.decision_reason || `ระบบพบปัญหาเรื่อง ${formatQualityReasons(data.quality_reasons)}` };
  if (data.final_label === 'unknown_cat') return { title: 'แมวที่ไม่อยู่ในระบบ', copy: data.decision_reason || 'ยังไม่พบแมวที่ตรงกับฐานข้อมูลในระดับที่มั่นใจได้' };
  return { title: formatLabel(data.final_label), copy: data.decision_reason || 'ระบบพบข้อมูลแมวที่ตรงกับฐานข้อมูล' };
}

function hideResultCard() { resultState.classList.remove('hidden'); resultCard.classList.add('hidden'); }
function setResultState(message) { hideResultCard(); resultState.textContent = message; }
function showResultCard() { resultState.classList.add('hidden'); resultCard.classList.remove('hidden'); }

function resetRenderedResult(message = 'พร้อม') {
  resetAnalysisPlayback();
  hideResultCard();
  resultState.textContent = message;
  finalLabel.textContent = 'ยังไม่มีผลลัพธ์';
  scorePill.textContent = 'score -';
  analysisPanel?.classList.add('hidden');
  analysisPreview?.removeAttribute('src');
  analysisCropBox?.classList.add('hidden');
  analysisCropBox?.removeAttribute('style');
  analysisFaceBox?.classList.add('hidden');
  analysisFaceBox?.removeAttribute('style');
  modelInputPanel?.classList.add('hidden');
  modelInputPreview?.removeAttribute('src');
  profileDetails.classList.add('hidden');
  statusDetails.classList.add('hidden');
  majorityVotePanel.classList.add('hidden');
  [analysisSummary, analysisNote, modelInputMeta, modelInputNote, resultSimilarity, resultCatName, resultOwner, resultContact, resultLocation, statusTitle, statusCopy, bestKnown, secondScore, bestUnknown, bestNotCat, qualityNotes, decisionSummaryText, winnerLabel, winnerVotes, winnerTop10Votes, winnerWeightedSum, runnerUpLabel, decisionReason].filter(Boolean).forEach((el) => { el.textContent = '-'; });
  topCandidates.innerHTML = '';
  classSummaryList.innerHTML = '';
  resultFootnote.textContent = '';
  resultFootnote.classList.add('hidden');
  viewProfileBtn.classList.add('hidden');
  viewProfileBtn.href = '/cats';
  resultActions.classList.add('single');
}

function setPreview(dataUrl, label) {
  preview.src = dataUrl;
  preview.alt = label;
  fileName.textContent = label;
  predictBtn.disabled = false;
  resetRenderedResult();
}

function clearSelectedImage() {
  state.currentFile = null; state.capturedDataUrl = null; fileInput.value = '';
  preview.removeAttribute('src'); preview.alt = 'Preview'; fileName.textContent = 'ยังไม่มีรูป'; predictBtn.disabled = true;
}
function resetPredictionFlow() { clearSelectedImage(); resetRenderedResult(); }

function renderAnalysisPanel(data) {
  if (!analysisPanel || !analysisFrame || !analysisPreview || !analysisSummary || !analysisCropBox || !analysisFaceBox || !analysisSteps || !analysisNote) return;
  const imageWidth = Number(data.image_width ?? 0);
  const imageHeight = Number(data.image_height ?? 0);
  const faceBox = Array.isArray(data.face_box) ? data.face_box : null;
  const cropBox = Array.isArray(data.crop_box) ? data.crop_box : null;
  const hasLocalizedFace = Boolean(data.localized_face_detected && faceBox);
  const hasCatFaceSupport = Boolean(data.cat_face_model_supported);
  resetAnalysisPlayback();
  analysisPreview.src = preview.src;
  analysisPreview.alt = fileName.textContent || 'Analysis preview';
  analysisSummary.textContent = hasLocalizedFace ? 'สแกนทั้งภาพแล้วเจอหน้าแมว' : 'สแกนทั้งภาพก่อน แล้วค่อยส่ง crop เข้าโมเดล';
  setAbsoluteBoxStyle(analysisCropBox, cropBox, imageWidth, imageHeight);
  setAbsoluteBoxStyle(analysisFaceBox, faceBox, imageWidth, imageHeight);
  analysisCropBox.classList.add('hidden');
  analysisFaceBox.classList.add('hidden');
  analysisNote.textContent = hasLocalizedFace
    ? `ระบบสแกนทั้งภาพก่อน แล้วเจอกรอบหน้าแมวจริง จากนั้นขยายกรอบและส่งเฉพาะส่วนนั้นเข้าโมเดล`
    : hasCatFaceSupport
      ? `ระบบสแกนทั้งภาพแล้วไม่เจอกรอบหน้าแมวจริง จึงครอปกลางภาพส่งเข้าโมเดลต่อ ตอนนี้ face model แค่มองว่าคล้ายหน้าแมว ไม่ได้หมายถึงจับกรอบหน้าแมวได้`
      : `ระบบสแกนทั้งภาพแล้วไม่เจอกรอบหน้าแมว จากนั้นครอปกลางภาพส่งเข้าโมเดลต่อ`;
  analysisPanel.classList.remove('hidden');
  scheduleAnalysisStep(80, () => {
    analysisFrame.classList.add('play-scan');
    analysisSteps.querySelector('[data-step="scan"]')?.classList.add('active');
  });
  scheduleAnalysisStep(760, () => {
    analysisCropBox.classList.remove('hidden');
    analysisSteps.querySelector('[data-step="crop"]')?.classList.add('active');
  });
  scheduleAnalysisStep(1240, () => {
    if (hasLocalizedFace) {
      analysisFaceBox.classList.remove('hidden');
    }
    analysisSteps.querySelector('[data-step="model"]')?.classList.add('active');
  });
}

function renderClosestCandidatePreview(data) {
  if (!modelInputPanel || !modelInputPreview || !modelInputMeta || !modelInputNote) return;
  
  // พยายามหารูปที่ดีที่สุดของ "ตัวที่ชนะ" (final_label) ก่อน
  let closestCandidate = (data.top_candidates || []).find((row) => 
    row?.file_path && String(row.label) === String(data.final_label)
  );

  // ถ้าหาไม่เจอ (เช่น เป็น unknown_cat) ค่อยเอารูปที่คะแนนสูงสุดจริงๆ
  if (!closestCandidate) {
    closestCandidate = (data.top_candidates || []).find((row) => row?.file_path);
  }

  const imageUrl = uploadedFileUrl(closestCandidate?.file_path);
  if (!closestCandidate || !imageUrl) {
    modelInputPanel.classList.add('hidden');
    modelInputPreview.removeAttribute('src');
    modelInputMeta.textContent = '-';
    modelInputNote.textContent = '-';
    return;
  }
  modelInputPreview.src = imageUrl;
  modelInputPreview.alt = `รูปที่เหมือนที่สุด: ${formatLabel(closestCandidate.label)}`;
  modelInputMeta.textContent = `#${closestCandidate.rank ?? 1} · score ${formatScore(closestCandidate.score)}`;
  const candidateName = formatLabel(closestCandidate.label);
  const imageName = formatText(closestCandidate.image_name, '-');
  modelInputNote.textContent = `${candidateName} • ${formatText(closestCandidate.group, '-')} • ${imageName}`;
  modelInputPanel.classList.remove('hidden');
}

function renderMajorityVote(data) {
  majorityVotePanel.classList.remove('hidden');
  const decision = data.decision || {};
  const voteWindow = data.vote_window || {};
  const effectiveCount = Number(voteWindow.effective_candidate_count ?? 0);
  const topCount = Number(voteWindow.top_candidate_count ?? data.top_candidates?.length ?? 0);
  decisionSummaryText.textContent = `${formatLabel(data.final_label)} · ใช้โหวต ${effectiveCount}/${topCount} candidates`;
  winnerLabel.textContent = formatLabel(decision.winner_label);
  winnerVotes.textContent = String(decision.winner_votes ?? '-');
  winnerTop10Votes.textContent = String(decision.winner_top10_votes ?? '-');
  winnerWeightedSum.textContent = formatScore(decision.winner_weighted_sum);
  runnerUpLabel.textContent = formatLabel(decision.runner_up_label);
  const cutoffText = Number.isFinite(Number(voteWindow.cutoff_score))
    ? `ใช้เฉพาะ candidate ที่คะแนนไม่ต่ำกว่า ${formatScore(voteWindow.cutoff_score)}`
    : '';
  decisionReason.textContent = [data.decision_reason || '-', cutoffText].filter(Boolean).join(' • ');
  topCandidates.innerHTML = (data.top_candidates || []).map((row) => `
    <li class="candidate-row"><div><strong>#${row.rank}</strong><span>${escapeHtml(formatLabel(row.label))}</span></div><div class="candidate-meta"><small>${escapeHtml(row.group)}</small><strong>${formatScore(row.score)}</strong></div></li>
  `).join('') || '<li class="candidate-row"><div><strong>-</strong><span>ยังไม่มี candidate</span></div><div class="candidate-meta"><small>-</small><strong>-</strong></div></li>';
  classSummaryList.innerHTML = (data.top_class_summary || []).map((row) => `
    <li class="candidate-row compact"><div><span>${escapeHtml(formatLabel(row.label))}</span><small>${escapeHtml(row.group)}</small></div><div class="candidate-meta stacked"><strong>${row.votes} used · ${row.top10_votes ?? row.votes} top10</strong><small>used sum ${formatScore(row.weighted_sum)} · top10 sum ${formatScore(row.top10_weighted_sum ?? row.weighted_sum)}</small></div></li>
  `).join('') || '<li class="candidate-row compact"><div><span>ยังไม่มีสรุปคลาส</span></div></li>';
}

function renderProfileResult(data) {
  const cat = data.matched_cat || {};
  const margin = Math.max(Number(data.best_known_score || 0) - Number(data.second_known_score || 0), 0);
  profileDetails.classList.remove('hidden');
  statusDetails.classList.add('hidden');
  resultSimilarity.textContent = formatScore(data.best_known_score);
  resultCatName.textContent = formatText(cat.name, '-');
  resultOwner.textContent = formatText(cat.owner); resultContact.textContent = formatText(cat.contact); resultLocation.textContent = formatText(cat.location);
  resultFootnote.textContent = `ชนะ majority vote ด้วยคะแนน ${formatScore(data.best_known_score)} และห่างจาก known candidate รองลงมา ${formatScore(margin)}`;
  resultFootnote.classList.remove('hidden');
  viewProfileBtn.href = `/profile/${encodeURIComponent(cat.id)}`;
  viewProfileBtn.classList.remove('hidden');
  resultActions.classList.remove('single');
}

function renderStatusResult(data) {
  const summary = statusSummary(data);
  const qualityMessage = data.quality_pass ? `คุณภาพภาพผ่านเกณฑ์ • blur ${formatScore(data.blur_score, 2)} • light ${formatScore(data.brightness, 2)} • crop ${formatText(data.crop_strategy, '-')}` : `คุณภาพภาพไม่ผ่าน • ${formatQualityReasons(data.quality_reasons)} • blur ${formatScore(data.blur_score, 2)} • light ${formatScore(data.brightness, 2)}`;
  profileDetails.classList.add('hidden'); statusDetails.classList.remove('hidden');
  statusTitle.textContent = summary.title; statusCopy.textContent = summary.copy;
  bestKnown.textContent = `${formatText(data.best_known_name, '-')} · ${formatScore(data.best_known_score)}`;
  secondScore.textContent = formatScore(data.second_known_score); bestUnknown.textContent = formatScore(data.best_unknown_score); bestNotCat.textContent = formatScore(data.best_not_cat_score);
  qualityNotes.textContent = qualityMessage; resultFootnote.classList.add('hidden'); resultFootnote.textContent = ''; viewProfileBtn.classList.add('hidden'); resultActions.classList.add('single');
}

function renderResult(data) {
  showResultCard();
  finalLabel.textContent = data.matched_cat?.name || formatLabel(data.final_label);
  scorePill.textContent = `score ${formatScore(data.best_known_score)}`;
  renderAnalysisPanel(data);
  renderClosestCandidatePreview(data);
  renderMajorityVote(data);
  if (data.matched_cat) { renderProfileResult(data); return; }
  renderStatusResult(data);
}

async function predict() {
  predictBtn.disabled = true; setResultState('กำลังทำนาย...');
  let response;
  if (state.currentFile) {
    const formData = new FormData(); formData.append('file', state.currentFile); response = await fetch('/api/predict', { method: 'POST', body: formData });
  } else if (state.capturedDataUrl) {
    response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_base64: state.capturedDataUrl }) });
  } else { setResultState('เลือกรูปก่อน'); predictBtn.disabled = false; return; }
  const data = await readJson(response); predictBtn.disabled = false;
  if (!response.ok) { setResultState(data.message || 'ทำนายไม่สำเร็จ'); return; }
  renderResult(data);
}

function stopCamera() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.srcObject = null;
}

async function startCamera() {
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: state.cameraFacingMode } },
      audio: false,
    });
  } catch (_error) {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  video.srcObject = state.stream;
  video.classList.toggle('mirror', state.cameraFacingMode === 'user');
}

async function switchCamera() {
  state.cameraFacingMode = state.cameraFacingMode === 'environment' ? 'user' : 'environment';
  await startCamera();
}

function captureFrame() {
  if (!state.stream) return;
  const width = video.videoWidth || 640, height = video.videoHeight || 480;
  cameraCanvas.width = width; cameraCanvas.height = height;
  const ctx = cameraCanvas.getContext('2d'); ctx.drawImage(video, 0, 0, width, height);
  state.capturedDataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95); state.currentFile = null; setPreview(state.capturedDataUrl, 'camera_capture.jpg');
}

fileInput.addEventListener('change', (event) => { const [file] = event.target.files; if (!file) return; state.currentFile = file; state.capturedDataUrl = null; const reader = new FileReader(); reader.onload = () => setPreview(reader.result, file.name); reader.readAsDataURL(file); });
predictBtn.addEventListener('click', predict); retryPredictBtn.addEventListener('click', resetPredictionFlow); startCameraBtn.addEventListener('click', startCamera); switchCameraBtn.addEventListener('click', switchCamera); captureBtn.addEventListener('click', captureFrame);
document.querySelectorAll('.tab-pill').forEach((button) => { button.addEventListener('click', () => { document.querySelectorAll('.tab-pill').forEach((tab) => tab.classList.remove('active')); document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active')); button.classList.add('active'); document.getElementById(`tab-${button.dataset.tab}`).classList.add('active'); }); });
window.addEventListener('beforeunload', stopCamera);
resetRenderedResult(); refreshSummary();
