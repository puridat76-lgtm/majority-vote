const {
  escapeHtml,
  readJson,
  refreshSummary,
  setMessage,
  updateGlobalStatus,
} = window.CatIdentity;

const catList = document.getElementById('catList');
const catsEmpty = document.getElementById('catsEmpty');
const catSummaryText = document.getElementById('catSummaryText');
const searchInput = document.getElementById('searchInput');
const newCatBtn = document.getElementById('newCatBtn');
const catForm = document.getElementById('catForm');
const catIdInput = document.getElementById('catIdInput');
const formModeLabel = document.getElementById('formModeLabel');
const formTitle = document.getElementById('formTitle');
const nameInput = document.getElementById('nameInput');
const ownerInput = document.getElementById('ownerInput');
const contactInput = document.getElementById('contactInput');
const locationInput = document.getElementById('locationInput');
const uploadBox = document.getElementById('uploadBox');
const imagesInput = document.getElementById('imagesInput');
const imagesHint = document.getElementById('imagesHint');
const uploadPreviewGrid = document.getElementById('uploadPreviewGrid');
const existingImageCount = document.getElementById('existingImageCount');
const imageGrid = document.getElementById('imageGrid');
const formMessage = document.getElementById('formMessage');
const deleteCatBtn = document.getElementById('deleteCatBtn');
const resetFormBtn = document.getElementById('resetFormBtn');
const catCameraVideo = document.getElementById('catCameraVideo');
const catCameraCanvas = document.getElementById('catCameraCanvas');
const catStartCameraBtn = document.getElementById('catStartCameraBtn');
const catSwitchCameraBtn = document.getElementById('catSwitchCameraBtn');
const catCaptureBtn = document.getElementById('catCaptureBtn');

const state = {
  cats: [],
  selectedCatId: null,
  search: '',
  pendingImagePreviews: [],
  uploadWarnings: [],
  uploadSizeError: '',
  cameraStream: null,
  cameraFacingMode: 'environment',
};

let preferredCatId = Number(new URLSearchParams(window.location.search).get('cat_id')) || null;
const maxUploadBytes = Number(uploadBox?.dataset.maxUploadBytes) || 64 * 1024 * 1024;

function selectedCat() {
  return state.cats.find((cat) => cat.id === state.selectedCatId) || null;
}

function clearPendingImagePreviews() {
  state.pendingImagePreviews.forEach((preview) => URL.revokeObjectURL(preview.url));
  state.pendingImagePreviews = [];
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  renderUploadPreviewGrid();
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${Math.ceil(bytes / 1024)} KB`;
}

function formatDuplicateWarnings(warnings) {
  return warnings.map((warning) => {
    if (warning.source === 'upload_batch') {
      return `${warning.filename} ซ้ำกับไฟล์ ${warning.duplicate_of} ที่เลือกพร้อมกัน`;
    }
    return `${warning.filename} ซ้ำกับรูปของ ${warning.cat_name} (${warning.duplicate_of})`;
  }).join('; ');
}

function previewStatusClass(image) {
  if (image.status === 'duplicate') {
    return 'duplicate';
  }
  if (image.status === 'unique') {
    return 'unique';
  }
  if (image.status === 'checking') {
    return 'checking';
  }
  return '';
}

function previewStatusLabel(image) {
  if (image.status === 'duplicate') {
    return 'ซ้ำ';
  }
  if (image.status === 'unique') {
    return 'ไม่ซ้ำ';
  }
  if (image.status === 'checking') {
    return 'กำลังตรวจ';
  }
  return 'รอบันทึก';
}

function syncPendingFilesToInput() {
  const transfer = new DataTransfer();
  state.pendingImagePreviews.forEach((preview) => {
    if (preview.file) {
      transfer.items.add(preview.file);
    }
  });
  imagesInput.files = transfer.files;
  imagesHint.textContent = `${state.pendingImagePreviews.length} ไฟล์`;
}

function appendPendingFiles(files) {
  files.forEach((file) => {
    state.pendingImagePreviews.push({
      name: file.name,
      file,
      status: 'checking',
      duplicateDetail: '',
      url: URL.createObjectURL(file),
    });
  });
  syncPendingFilesToInput();
  renderUploadPreviewGrid();
  renderImageGrid(selectedCat()?.images || []);
}

function stopCatCamera() {
  if (!state.cameraStream) return;
  state.cameraStream.getTracks().forEach((track) => track.stop());
  state.cameraStream = null;
  catCameraVideo.srcObject = null;
}

async function startCatCamera() {
  stopCatCamera();
  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: state.cameraFacingMode } },
      audio: false,
    });
  } catch (_error) {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  catCameraVideo.srcObject = state.cameraStream;
  catCameraVideo.classList.toggle('mirror', state.cameraFacingMode === 'user');
}

async function switchCatCamera() {
  state.cameraFacingMode = state.cameraFacingMode === 'environment' ? 'user' : 'environment';
  await startCatCamera();
}

function canvasToFile(canvas, filename) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(blob ? new File([blob], filename, { type: 'image/jpeg' }) : null);
    }, 'image/jpeg', 0.95);
  });
}

async function captureCatCameraImage() {
  if (!state.cameraStream) return;
  const width = catCameraVideo.videoWidth || 640;
  const height = catCameraVideo.videoHeight || 480;
  catCameraCanvas.width = width;
  catCameraCanvas.height = height;
  catCameraCanvas.getContext('2d').drawImage(catCameraVideo, 0, 0, width, height);
  const file = await canvasToFile(catCameraCanvas, `cat_camera_${Date.now()}.jpg`);
  if (!file) return;
  appendPendingFiles([file]);
  setMessage(formMessage, 'กำลังตรวจรูปซ้ำ...', 'neutral');
  await validateSelectedImages();
}

async function removePendingImage(index) {
  const [removed] = state.pendingImagePreviews.splice(index, 1);
  if (removed?.url) {
    URL.revokeObjectURL(removed.url);
  }
  syncPendingFilesToInput();
  renderUploadPreviewGrid();
  renderImageGrid(selectedCat()?.images || []);
  if (state.pendingImagePreviews.length) {
    await validateSelectedImages();
    return;
  }
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  setMessage(formMessage, '');
}

async function fileHash(file) {
  const digest = await window.crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function existingImageHashIndex() {
  const index = new Map();
  state.cats.forEach((cat) => {
    (cat.images || []).forEach((image) => {
      if (!image.content_hash || index.has(image.content_hash)) {
        return;
      }
      index.set(image.content_hash, {
        catName: cat.name,
        imageName: image.source_name || image.name,
      });
    });
  });
  return index;
}

async function validateSelectedImages() {
  const files = [...imagesInput.files];
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  state.pendingImagePreviews.forEach((preview) => {
    preview.status = files.length ? 'checking' : 'pending';
    preview.duplicateDetail = '';
  });
  renderUploadPreviewGrid();
  renderImageGrid(selectedCat()?.images || []);

  if (!files.length) {
    setMessage(formMessage, '');
    return true;
  }

  if (!window.crypto?.subtle) {
    state.pendingImagePreviews.forEach((preview) => {
      preview.status = 'pending';
    });
    renderUploadPreviewGrid();
    renderImageGrid(selectedCat()?.images || []);
    setMessage(formMessage, 'เบราว์เซอร์นี้ยังตรวจรูปซ้ำก่อนอัปโหลดไม่ได้ ระบบจะตรวจอีกครั้งหลังบันทึก', 'neutral');
    return true;
  }

  const existingHashes = existingImageHashIndex();
  const seenUploads = new Map();
  for (const [index, file] of files.entries()) {
    const preview = state.pendingImagePreviews[index];
    const hash = await fileHash(file);
    if (preview) {
      preview.hash = hash;
      preview.status = 'unique';
      preview.duplicateDetail = '';
    }
    const existing = existingHashes.get(hash);
    if (existing) {
      if (preview) {
        preview.status = 'duplicate';
        preview.duplicateDetail = `ซ้ำกับรูปของ ${existing.catName} (${existing.imageName})`;
      }
      state.uploadWarnings.push({
        filename: file.name,
        duplicate_of: existing.imageName,
        cat_name: existing.catName,
        source: 'existing',
      });
      continue;
    }
    if (seenUploads.has(hash)) {
      const firstMatch = seenUploads.get(hash);
      if (preview) {
        preview.status = 'duplicate';
        preview.duplicateDetail = `ซ้ำกับไฟล์ ${firstMatch.name} ที่เลือกพร้อมกัน`;
      }
      const firstPreview = state.pendingImagePreviews[firstMatch.index];
      if (firstPreview) {
        firstPreview.status = 'duplicate';
        firstPreview.duplicateDetail = `ซ้ำกับไฟล์ ${file.name} ที่เลือกพร้อมกัน`;
      }
      state.uploadWarnings.push({
        filename: file.name,
        duplicate_of: firstMatch.name,
        cat_name: '',
        source: 'upload_batch',
      });
      continue;
    }
    seenUploads.set(hash, { name: file.name, index });
  }

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  if (totalBytes > maxUploadBytes) {
    state.uploadSizeError = `ไฟล์ที่เลือกมีขนาดรวม ${formatBytes(totalBytes)} เกิน limit ${formatBytes(maxUploadBytes)}`;
  }
  renderUploadPreviewGrid();
  renderImageGrid(selectedCat()?.images || []);

  if (state.uploadWarnings.length) {
    setMessage(formMessage, `พบรูปซ้ำ: ${formatDuplicateWarnings(state.uploadWarnings)}`, 'error');
    return false;
  }
  if (state.uploadSizeError) {
    setMessage(formMessage, state.uploadSizeError, 'error');
    return false;
  }

  setMessage(formMessage, '');
  return true;
}

function renderUploadPreviewGrid() {
  if (!uploadPreviewGrid) {
    return;
  }
  if (!state.pendingImagePreviews.length) {
    uploadPreviewGrid.innerHTML = '';
    uploadPreviewGrid.classList.add('hidden');
    return;
  }

  uploadPreviewGrid.classList.remove('hidden');
  uploadPreviewGrid.innerHTML = state.pendingImagePreviews
    .map((image, index) => `
        <article class="upload-preview-card ${previewStatusClass(image)}" title="${escapeHtml(image.duplicateDetail || image.name)}">
          <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" />
          <span class="preview-status-badge">${previewStatusLabel(image)}</span>
          <button type="button" class="pending-remove-btn" data-remove-pending-image="${index}">เอาออก</button>
        </article>
      `)
    .join('');
}

function renderImageGrid(images, pendingImages = state.pendingImagePreviews) {
  const totalImages = images.length + pendingImages.length;
  existingImageCount.textContent = `${totalImages}`;
  if (!totalImages) {
    imageGrid.innerHTML = '<div class="empty-inline">ยังไม่มีรูป</div>';
    return;
  }

  const savedCards = images
    .map((image) => `
        <article class="thumb-card">
          <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" />
          <button type="button" class="thumb-delete" data-image-name="${escapeHtml(image.name)}">ลบ</button>
        </article>
      `)
    .join('');

  const pendingCards = pendingImages
    .map((image, index) => `
        <article class="thumb-card pending ${previewStatusClass(image)}" title="${escapeHtml(image.duplicateDetail || image.name)}">
          <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" />
          <span class="thumb-badge">${previewStatusLabel(image)}</span>
          <button type="button" class="pending-remove-btn" data-remove-pending-image="${index}">เอาออก</button>
        </article>
      `)
    .join('');

  imageGrid.innerHTML = savedCards + pendingCards;
}

function resetForm() {
  clearPendingImagePreviews();
  state.selectedCatId = null;
  catIdInput.value = '';
  nameInput.value = '';
  ownerInput.value = '';
  contactInput.value = '';
  locationInput.value = '';
  imagesInput.value = '';
  imagesHint.textContent = '0 ไฟล์';
  formModeLabel.textContent = 'เพิ่ม';
  formTitle.textContent = 'รายการใหม่';
  deleteCatBtn.classList.add('hidden');
  renderImageGrid([]);
  setMessage(formMessage, '');
  renderCats();
}

function fillForm(cat) {
  clearPendingImagePreviews();
  state.selectedCatId = cat.id;
  catIdInput.value = cat.id;
  nameInput.value = cat.name;
  ownerInput.value = cat.owner || '';
  contactInput.value = cat.contact || '';
  locationInput.value = cat.location || '';
  imagesInput.value = '';
  imagesHint.textContent = '0 ไฟล์';
  formModeLabel.textContent = 'แก้ไข';
  formTitle.textContent = cat.name;
  deleteCatBtn.classList.remove('hidden');
  renderImageGrid(cat.images || []);
  setMessage(formMessage, '');
  renderCats();
}

function renderCats() {
  const filtered = state.cats.filter((cat) => {
    if (!state.search) {
      return true;
    }
    const text = [cat.name, cat.owner, cat.contact, cat.location].join(' ').toLowerCase();
    return text.includes(state.search);
  });

  catSummaryText.textContent = `${state.cats.length} ตัว`;
  catsEmpty.classList.toggle('hidden', filtered.length !== 0);
  catList.innerHTML = filtered
    .map((cat) => {
      const selected = cat.id === state.selectedCatId ? 'selected' : '';
      const meta = [cat.owner, cat.contact, cat.location].filter(Boolean).join(' • ') || 'ยังไม่ระบุ';
      const cover = cat.cover_image
        ? `<img src="${cat.cover_image}" alt="${escapeHtml(cat.name)}" />`
        : `<div class="thumb-fallback">${escapeHtml(cat.name.slice(0, 1).toUpperCase())}</div>`;

      return `
        <article class="list-item ${selected}">
          <button type="button" class="item-main" data-select-cat="${cat.id}">
            <div class="item-cover">${cover}</div>
            <div class="item-copy">
              <strong>${escapeHtml(cat.name)}</strong>
              <span>${escapeHtml(meta)}</span>
            </div>
          </button>
          <div class="item-side">
            <span class="count-pill">${cat.image_count} ภาพ</span>
            <div class="inline-actions">
              <button type="button" class="text-btn" data-select-cat="${cat.id}">แก้ไข</button>
              <button type="button" class="text-btn danger" data-delete-cat="${cat.id}">ลบ</button>
            </div>
          </div>
        </article>
      `;
    })
    .join('');
}

async function loadCats({ selectFirst = false } = {}) {
  const response = await fetch('/api/cats');
  const data = await readJson(response);
  if (!response.ok) {
    return;
  }

  state.cats = data.cats || [];
  updateGlobalStatus(data.summary);

  if (preferredCatId && state.cats.some((cat) => cat.id === preferredCatId)) {
    state.selectedCatId = preferredCatId;
    preferredCatId = null;
  }

  if (!state.cats.some((cat) => cat.id === state.selectedCatId)) {
    state.selectedCatId = selectFirst && state.cats.length ? state.cats[0].id : null;
  }

  renderCats();
  const current = selectedCat();
  if (current) {
    fillForm(current);
  } else {
    resetForm();
  }
}

async function saveCat(event) {
  event.preventDefault();
  setMessage(formMessage, '');
  if (!(await validateSelectedImages())) {
    return;
  }

  const formData = new FormData();
  formData.append('name', nameInput.value.trim());
  formData.append('owner', ownerInput.value.trim());
  formData.append('contact', contactInput.value.trim());
  formData.append('location', locationInput.value.trim());
  [...imagesInput.files].forEach((file) => formData.append('images', file));

  const isEdit = Boolean(state.selectedCatId);
  const url = isEdit ? `/api/cats/${state.selectedCatId}` : '/api/cats';
  const method = isEdit ? 'PUT' : 'POST';

  const response = await fetch(url, { method, body: formData });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'บันทึกไม่สำเร็จ', 'error');
    return;
  }

  state.selectedCatId = data.cat.id;
  await Promise.all([loadCats(), refreshSummary()]);
  const current = selectedCat();
  if (current) {
    fillForm(current);
  }
  setMessage(formMessage, 'บันทึกแล้ว', 'success');
}

async function deleteCat(catId) {
  const cat = state.cats.find((row) => row.id === catId);
  if (!window.confirm(`ลบ ${cat?.name || 'รายการนี้'} ?`)) {
    return;
  }

  const response = await fetch(`/api/cats/${catId}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'ลบไม่สำเร็จ', 'error');
    return;
  }

  if (state.selectedCatId === catId) {
    state.selectedCatId = null;
  }
  await Promise.all([loadCats({ selectFirst: true }), refreshSummary()]);
  setMessage(formMessage, 'ลบแล้ว', 'success');
}

async function deleteImage(imageName) {
  if (!state.selectedCatId || !window.confirm('ลบรูปนี้?')) {
    return;
  }

  const response = await fetch(`/api/cats/${state.selectedCatId}/images/${encodeURIComponent(imageName)}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'ลบรูปไม่สำเร็จ', 'error');
    return;
  }

  await Promise.all([loadCats(), refreshSummary()]);
  const current = selectedCat();
  if (current) {
    fillForm(current);
  }
  setMessage(formMessage, 'ลบรูปแล้ว', 'success');
}

imagesInput.addEventListener('change', async () => {
  clearPendingImagePreviews();
  appendPendingFiles([...imagesInput.files]);
  if (state.pendingImagePreviews.length) {
    setMessage(formMessage, 'กำลังตรวจรูปซ้ำ...', 'neutral');
  }
  await validateSelectedImages();
});

searchInput.addEventListener('input', (event) => {
  state.search = event.target.value.trim().toLowerCase();
  renderCats();
});

newCatBtn.addEventListener('click', resetForm);
resetFormBtn.addEventListener('click', resetForm);
catForm.addEventListener('submit', saveCat);
catStartCameraBtn.addEventListener('click', startCatCamera);
catSwitchCameraBtn.addEventListener('click', switchCatCamera);
catCaptureBtn.addEventListener('click', captureCatCameraImage);

catList.addEventListener('click', (event) => {
  const selectButton = event.target.closest('[data-select-cat]');
  if (selectButton) {
    const cat = state.cats.find((row) => row.id === Number(selectButton.dataset.selectCat));
    if (cat) {
      fillForm(cat);
    }
    return;
  }

  const deleteButton = event.target.closest('[data-delete-cat]');
  if (deleteButton) {
    deleteCat(Number(deleteButton.dataset.deleteCat));
  }
});

imageGrid.addEventListener('click', (event) => {
  const removePendingButton = event.target.closest('[data-remove-pending-image]');
  if (removePendingButton) {
    removePendingImage(Number(removePendingButton.dataset.removePendingImage));
    return;
  }

  const deleteButton = event.target.closest('[data-image-name]');
  if (deleteButton) {
    deleteImage(deleteButton.dataset.imageName);
  }
});

uploadPreviewGrid?.addEventListener('click', (event) => {
  const removePendingButton = event.target.closest('[data-remove-pending-image]');
  if (removePendingButton) {
    event.preventDefault();
    removePendingImage(Number(removePendingButton.dataset.removePendingImage));
  }
});

deleteCatBtn.addEventListener('click', () => {
  if (state.selectedCatId) {
    deleteCat(state.selectedCatId);
  }
});

window.addEventListener('beforeunload', () => {
  clearPendingImagePreviews();
  stopCatCamera();
});

resetForm();
loadCats({ selectFirst: true });
