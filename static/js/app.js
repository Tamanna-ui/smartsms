const messageInput = document.getElementById("message");
const predictionForm = document.getElementById("prediction-form");
const batchForm = document.getElementById("batch-form");
const batchFile = document.getElementById("batch-file");
const filterButtons = document.querySelectorAll(".filter-chip");
const messageList = document.getElementById("message-list");
const batchSummary = document.getElementById("batch-summary");
const batchResults = document.getElementById("batch-results");

const predictionCard = document.getElementById("prediction-card");
const predictionStatus = document.getElementById("prediction-status");
const predictionDot = document.getElementById("prediction-dot");
const predictionConfidence = document.getElementById("prediction-confidence");
const predictionMessage = document.getElementById("prediction-message");
const predictionExplanation = document.getElementById("prediction-explanation");
const termList = document.getElementById("term-list");

const statsTotal = document.getElementById("stat-total");
const statsRate = document.getElementById("stat-rate");
const spamCount = document.getElementById("spam-count");
const hamCount = document.getElementById("ham-count");
const reviewCount = document.getElementById("review-count");
const avgConfidence = document.getElementById("avg-confidence");
const modelName = document.getElementById("model-name");
const datasetSize = document.getElementById("dataset-size");
const testSize = document.getElementById("test-size");

function renderTerms(spamTerms, safeTerms) {
  termList.innerHTML = "";

  if ((!Array.isArray(spamTerms) || spamTerms.length === 0) && (!Array.isArray(safeTerms) || safeTerms.length === 0)) {
    return;
  }

  (spamTerms || []).forEach((term) => {
    const chip = document.createElement("span");
    chip.className = "term-chip term-chip-spam";
    chip.textContent = term;
    termList.appendChild(chip);
  });

  (safeTerms || []).forEach((term) => {
    const chip = document.createElement("span");
    chip.className = "term-chip term-chip-safe";
    chip.textContent = term;
    termList.appendChild(chip);
  });
}

function renderPrediction(result) {
  const label = result.label || "unknown";
  predictionCard.className = `prediction-card prediction-${label}`;
  predictionStatus.textContent = result.status || "Unknown";
  predictionDot.className = `status-dot status-dot-${label}`;
  predictionConfidence.textContent = `${Number(result.confidence || 0).toFixed(2)}%`;
  predictionMessage.textContent = result.message || "Prediction unavailable.";
  predictionExplanation.textContent = result.explanation || "No explanation available.";
  renderTerms(result.suspicious_terms || [], result.safe_terms || []);
}

function renderMessages(messages) {
  messageList.innerHTML = "";

  if (!Array.isArray(messages) || messages.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.textContent = "No messages found for this filter.";
    messageList.appendChild(emptyState);
    return;
  }

  messages.forEach((message) => {
    const card = document.createElement("article");
    card.className = `message-card message-card-${message.label}`;
    card.innerHTML = `
      <div class="message-topline">
        <span class="message-badge">${message.status}</span>
        <span class="message-confidence">${Number(message.confidence || 0).toFixed(2)}%</span>
      </div>
      <p class="message-body"></p>
      <p class="message-meta">${message.source} • ${message.created_at}</p>
    `;

    card.querySelector(".message-body").textContent = message.body;
    messageList.appendChild(card);
  });
}

function updateHistoryStats(history) {
  if (!history) {
    return;
  }

  statsTotal.textContent = history.total_messages;
  statsRate.textContent = `${Number(history.spam_rate || 0).toFixed(2)}%`;
  spamCount.textContent = history.spam_count;
  hamCount.textContent = history.ham_count;
  reviewCount.textContent = history.review_count || 0;
  avgConfidence.textContent = `${Number(history.average_confidence || 0).toFixed(2)}%`;
}

function renderBatchResults(payload) {
  const summary = payload.summary || {};
  batchSummary.innerHTML = `
    <div class="stats-grid">
      <article class="stat-card"><span>Total</span><strong>${summary.total || 0}</strong></article>
      <article class="stat-card"><span>Spam</span><strong>${summary.spam || 0}</strong></article>
      <article class="stat-card"><span>Safe</span><strong>${summary.ham || 0}</strong></article>
      <article class="stat-card"><span>Needs Review</span><strong>${summary.review || 0}</strong></article>
    </div>
  `;

  const rows = (payload.results || []).slice(0, 20);
  batchResults.innerHTML = "";

  if (rows.length === 0) {
    batchResults.innerHTML = `<div class="empty-state">No messages were found in that CSV.</div>`;
    return;
  }

  rows.forEach((item) => {
    const card = document.createElement("article");
    card.className = `message-card message-card-${item.label}`;
    card.innerHTML = `
      <div class="message-topline">
        <span class="message-badge">${item.status}</span>
        <span class="message-confidence">${Number(item.confidence || 0).toFixed(2)}%</span>
      </div>
      <p class="message-body"></p>
      <p class="message-meta"></p>
    `;
    card.querySelector(".message-body").textContent = item.message;
    card.querySelector(".message-meta").textContent = item.explanation;
    batchResults.appendChild(card);
  });
}

async function refreshStats() {
  const response = await fetch("/api/stats");
  const payload = await response.json();
  updateHistoryStats(payload.history);

  if (payload.model) {
    modelName.textContent = payload.model.best_model || "Pending";
    datasetSize.textContent = payload.model.dataset_size ?? "--";
    testSize.textContent = payload.model.test_size ?? "--";
  }
}

async function loadMessages(filter = "all") {
  const response = await fetch(`/api/messages?filter=${encodeURIComponent(filter)}`);
  const payload = await response.json();
  renderMessages(payload.messages || []);
}

filterButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    filterButtons.forEach((chip) => chip.classList.remove("active"));
    button.classList.add("active");
    await loadMessages(button.dataset.filter || "all");
  });
});

if (predictionForm) {
  predictionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        source: "dashboard",
      }),
    });

    const result = await response.json();
    renderPrediction(result);
    updateHistoryStats(result.stats);

    const activeFilter = document.querySelector(".filter-chip.active")?.dataset.filter || "all";
    await loadMessages(activeFilter);
  });
}

if (batchForm) {
  batchForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!batchFile.files || batchFile.files.length === 0) {
      batchSummary.innerHTML = `<div class="empty-state">Choose a CSV file first.</div>`;
      batchResults.innerHTML = "";
      return;
    }

    const formData = new FormData();
    formData.append("file", batchFile.files[0]);

    const response = await fetch("/api/batch-predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      batchSummary.innerHTML = `<div class="empty-state">${payload.error || "Batch prediction failed."}</div>`;
      batchResults.innerHTML = "";
      return;
    }

    renderBatchResults(payload);
  });
}

renderMessages(window.appState?.initialMessages || []);
refreshStats();
