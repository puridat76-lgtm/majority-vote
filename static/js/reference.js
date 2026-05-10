const {
  escapeHtml,
  readJson,
  refreshSummary,
  setMessage,
  updateGlobalStatus,
} = window.CatIdentity;

const referencePage = document.getElementById('referencePage');
const referenceKey = referencePage.dataset.referenceKey;
const maxUploadBytes = Number(referencePage.dataset.maxUploadBytes) || 64 * 1024 * 1024;
const referenceInput = document.getElementById('referenceInput');
const referenceInputHint = document.getElementById('referenceInputHint');
const referencePendingCount = document.getElementById('referencePendingCount');
const referenceUploadPreviewGrid = document.getElementById('referenceUploadPreviewGrid');
const referenceSaveBtn = document.getElementById('referenceSaveBtn');
const referenceResetBtn = document.getElementById('referenceResetBtn');
const referenceCount = document.getElementById('referenceCount');
const referenceIndexState = document.getElementById('referenceIndexState');
const referenceVisibleCount = document.getElementById('referenceVisibleCount');
const referenceLoadMoreBtn = document.getElementById('referenceLoadMoreBtn');
const referenceGrid = document.getElementById('referenceGrid');
const referenceEmpty = document.getElementById('referenceEmpty');
const referenceMessage = document.getElementById('referenceMessage');

const state = {
  limit: 24,
  step: 128,
  pendingImagePreviews: [],
  uploadWarnings: [],
  uploadSizeError: '',
  duplicateHashIndex: {},
};

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

function releasePendingImagePreviews() {
  state.pendingImagePreviews.forEach((preview) => {
    if (preview.url) {
      URL.revokeObjectURL(preview.url);
    }
  });
}

function syncPendingFilesToInput() {
  const transfer = new DataTransfer();
  state.pendingImagePreviews.forEach((preview) => {
    if (preview.file) {
      transfer.items.add(preview.file);
    }
  });
  referenceInput.files = transfer.files;
  referenceInputHint.textContent = `${state.pendingImagePreviews.length} ไฟล์`;
  referencePendingCount.textContent = `${state.pendingImagePreviews.length} ไฟล์`;
  referenceSaveBtn.disabled = state.pendingImagePreviews.length === 0;
}

function clearPendingImagePreviews() {
  releasePendingImagePreviews();
  state.pendingImagePreviews = [];
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  referenceInput.value = '';
  syncPendingFilesToInput();
  renderUploadPreviewGrid();
}

function existingImageHashIndex() {
  return new Map(Object.entries(state.duplicateHashIndex || {}));
}

async function fileHash(file) {
  const digest = await window.crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function renderUploadPreviewGrid() {
  if (!state.pendingImagePreviews.length) {
    referenceUploadPreviewGrid.innerHTML = '';
    referenceUploadPreviewGrid.classList.add('hidden');
    return;
  }

  referenceUploadPreviewGrid.classList.remove('hidden');
  referenceUploadPreviewGrid.innerHTML = state.pendingImagePreviews
    .map((image, index) => `
      <article class="upload-preview-card ${previewStatusClass(image)}" title="${escapeHtml(image.duplicateDetail || image.name)}">
        <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.name)}" />
        <span class="preview-status-badge">${previewStatusLabel(image)}</span>
        <button type="button" class="pending-remove-btn" data-remove-pending-image="${index}">เอาออก</button>
      </article>
    `)
    .join('');
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
}

function renderReferenceGrid(referenceSet) {
  const images = referenceSet.images || [];
  const totalImages = referenceSet.image_count ?? images.length;
  const hiddenCount = referenceSet.hidden_count ?? 0;

  referenceVisibleCount.textContent = hiddenCount ? `${images.length} / ${totalImages} รูป` : `${totalImages} รูป`;
  referenceLoadMoreBtn.textContent = `+${Math.min(hiddenCount, state.step)} เพิ่มเติม`;
  referenceLoadMoreBtn.classList.toggle('hidden', hiddenCount === 0);
  referenceEmpty.classList.toggle('hidden', images.length !== 0);
  referenceGrid.innerHTML = images
    .map((image) => `
      <article class="thumb-card large">
        <img src="${image.url}" alt="${escapeHtml(image.source_name || image.name)}" />
        <button type="button" class="thumb-delete" data-image-name="${escapeHtml(image.name)}" title="ลบรูปนี้">ลบรูป</button>
      </article>
    `)
    .join('');
}

function renderReference(referenceSet, summary) {
  if (referenceCount) {
    referenceCount.textContent = referenceSet.image_count ?? 0;
  }
  if (referenceIndexState) {
    referenceIndexState.textContent = summary?.index_status === 'needs_train' ? 'ต้อง Train ใหม่' : 'พร้อมใช้งาน';
  }
  renderReferenceGrid(referenceSet);
}

function applyServerDuplicateWarnings(warnings) {
  if (!Array.isArray(warnings) || !warnings.length) {
    return;
  }
  state.uploadWarnings = warnings;
  state.pendingImagePreviews.forEach((preview) => {
    preview.status = 'unique';
    preview.duplicateDetail = '';
  });
  warnings.forEach((warning) => {
    const preview = state.pendingImagePreviews.find((item) => item.name === warning.filename);
    if (!preview) {
      return;
    }
    preview.status = 'duplicate';
    if (warning.source === 'upload_batch') {
      preview.duplicateDetail = `ซ้ำกับไฟล์ ${warning.duplicate_of} ที่เลือกพร้อมกัน`;
      return;
    }
    preview.duplicateDetail = `ซ้ำกับรูปของ ${warning.cat_name} (${warning.duplicate_of})`;
  });
  renderUploadPreviewGrid();
}

async function validateSelectedImages() {
  const files = [...referenceInput.files];
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  state.pendingImagePreviews.forEach((preview) => {
    preview.status = files.length ? 'checking' : 'pending';
    preview.duplicateDetail = '';
  });
  renderUploadPreviewGrid();

  if (!files.length) {
    setMessage(referenceMessage, '');
    return true;
  }

  if (!window.crypto?.subtle) {
    state.pendingImagePreviews.forEach((preview) => {
      preview.status = 'pending';
    });
    renderUploadPreviewGrid();
    setMessage(referenceMessage, 'เบราว์เซอร์นี้ยังตรวจรูปซ้ำก่อนอัปโหลดไม่ได้ ระบบจะตรวจอีกครั้งหลังบันทึก', 'neutral');
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
        preview.duplicateDetail = `ซ้ำกับรูปของ ${existing.cat_name} (${existing.image_name})`;
      }
      state.uploadWarnings.push({
        filename: file.name,
        duplicate_of: existing.image_name,
        cat_name: existing.cat_name,
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

  if (state.uploadWarnings.length) {
    setMessage(referenceMessage, `พบรูปซ้ำ: ${formatDuplicateWarnings(state.uploadWarnings)}`, 'error');
    return false;
  }
  if (state.uploadSizeError) {
    setMessage(referenceMessage, state.uploadSizeError, 'error');
    return false;
  }

  setMessage(referenceMessage, '');
  return true;
}

async function removePendingImage(index) {
  const [removed] = state.pendingImagePreviews.splice(index, 1);
  if (removed?.url) {
    URL.revokeObjectURL(removed.url);
  }
  syncPendingFilesToInput();
  renderUploadPreviewGrid();
  if (state.pendingImagePreviews.length) {
    await validateSelectedImages();
    return;
  }
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  setMessage(referenceMessage, '');
}

async function loadReference(limit = state.limit) {
  state.limit = limit;
  const response = await fetch(`/api/reference-sets/${referenceKey}?limit=${state.limit}`);
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(referenceMessage, data.message || 'โหลดข้อมูลไม่สำเร็จ', 'error');
    return;
  }

  state.duplicateHashIndex = data.duplicate_hash_index || {};
  renderReference(data.reference_set, data.summary);
  updateGlobalStatus(data.summary);
  if (state.pendingImagePreviews.length) {
    await validateSelectedImages();
  }
}

async function uploadImages() {
  if (!referenceInput.files.length) {
    setMessage(referenceMessage, 'เลือกรูปก่อน', 'error');
    return;
  }
  if (!(await validateSelectedImages())) {
    return;
  }

  const formData = new FormData();
  [...referenceInput.files].forEach((file) => formData.append('images', file));

  const response = await fetch(`/api/reference-sets/${referenceKey}/images`, { method: 'POST', body: formData });
  const data = await readJson(response);
  if (!response.ok) {
    applyServerDuplicateWarnings(data.duplicate_images);
    setMessage(referenceMessage, data.message || 'เพิ่มรูปไม่สำเร็จ', 'error');
    return;
  }

  clearPendingImagePreviews();
  await loadReference(state.limit);
  await refreshSummary();
  setMessage(referenceMessage, 'เพิ่มรูปแล้ว', 'success');
}

async function deleteImage(imageName) {
  if (!window.confirm('ลบรูปนี้?')) {
    return;
  }

  const response = await fetch(`/api/reference-sets/${referenceKey}/images/${encodeURIComponent(imageName)}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(referenceMessage, data.message || 'ลบรูปไม่สำเร็จ', 'error');
    return;
  }

  await loadReference(state.limit);
  await refreshSummary();
  setMessage(referenceMessage, 'ลบรูปแล้ว', 'success');
}

function loadMoreImages() {
  loadReference(state.limit + state.step);
}

referenceInput.addEventListener('change', async () => {
  const newFiles = [...referenceInput.files];
  if (!newFiles.length) {
    return;
  }
  appendPendingFiles(newFiles);
  state.uploadWarnings = [];
  state.uploadSizeError = '';
  syncPendingFilesToInput();
  renderUploadPreviewGrid();
  if (state.pendingImagePreviews.length) {
    setMessage(referenceMessage, 'กำลังตรวจรูปซ้ำ...', 'neutral');
  } else {
    setMessage(referenceMessage, '');
  }
  await validateSelectedImages();
});

referenceSaveBtn.addEventListener('click', uploadImages);
referenceResetBtn.addEventListener('click', () => {
  clearPendingImagePreviews();
  setMessage(referenceMessage, '');
});
referenceLoadMoreBtn.addEventListener('click', loadMoreImages);

referenceGrid.addEventListener('click', (event) => {
  const deleteButton = event.target.closest('[data-image-name]');
  if (deleteButton) {
    deleteImage(deleteButton.dataset.imageName);
  }
});

referenceUploadPreviewGrid.addEventListener('click', (event) => {
  const removePendingButton = event.target.closest('[data-remove-pending-image]');
  if (removePendingButton) {
    event.preventDefault();
    removePendingImage(Number(removePendingButton.dataset.removePendingImage));
  }
});

window.addEventListener('beforeunload', releasePendingImagePreviews);

clearPendingImagePreviews();
Promise.all([loadReference(), refreshSummary()]);
