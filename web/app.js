const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const rawEl = document.getElementById("raw");
const modelNameEl = document.getElementById("modelName");
const embedDimEl = document.getElementById("embedDim");

function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderClosest(title, metricLabel, metricValue, obj) {
  const idx = obj.index;
  const text = obj.text;

  return `
    <div class="sectionTitle">${esc(title)}</div>
    <div class="resultBox">
      <div class="kv">
        <div><span class="badge">rank 1</span></div>
        <div>Index: ${esc(idx)}</div>
        <div>${esc(metricLabel)}: ${esc(metricValue)}</div>
        <div>Matching text chunk:</div>
      </div>
      <div class="chunk">"${esc(text)}"</div>
    </div>
  `;
}

function renderTopList(title, items, metricKey, metricLabel, betterHint) {
  if (!Array.isArray(items) || items.length === 0) return "";

  const list = items
    .map((it) => {
      const metric = it[metricKey];
      return `
        <div class="listItem">
          <div class="kv">
            <div><span class="badge">rank ${esc(it.rank)}</span></div>
            <div>Index: ${esc(it.index)}</div>
            <div>${esc(metricLabel)}: ${esc(metric)}</div>
            <div>Chunk:</div>
          </div>
          <div class="chunk">"${esc(it.text)}"</div>
        </div>
      `;
    })
    .join("");

  return `
    <div class="sectionTitle">${esc(title)} <span style="color:#666;font-weight:700;font-size:12px;">${esc(betterHint)}</span></div>
    <div class="resultBox">
      ${list}
    </div>
  `;
}

async function doSearch() {
  const prompt = document.getElementById("prompt").value.trim();
  const top_k = Number(document.getElementById("topk").value);

  if (!prompt) {
    statusEl.textContent = "Please enter a prompt.";
    return;
  }

  resultsEl.innerHTML = "";
  rawEl.textContent = "";
  statusEl.textContent = "Searching...";
  btn.disabled = true;

  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, top_k })
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      statusEl.textContent = data.error ? String(data.error) : "Request failed.";
      btn.disabled = false;
      return;
    }

    rawEl.textContent = JSON.stringify(data, null, 2);

    modelNameEl.textContent = data.model ? data.model : "unknown";
    embedDimEl.textContent = data.embedding_dim ? String(data.embedding_dim) : "unknown";

    const euc = data.closest_euclidean;
    const cos = data.closest_cosine;

    let html = "";
    if (euc) {
      html += renderClosest(
        "CLOSEST MATCH (EUCLIDEAN)",
        "Distance",
        Number(euc.distance).toFixed(6),
        euc
      );
    }

    if (cos) {
      html += renderClosest(
        "CLOSEST MATCH (COSINE)",
        "Cosine Score",
        Number(cos.score).toFixed(6),
        cos
      );
    }

    html += renderTopList(
      `Top ${top_k} chunks (Euclidean)`,
      data.top_euclidean,
      "distance",
      "Distance",
      "lower is better"
    );

    html += renderTopList(
      `Top ${top_k} chunks (Cosine)`,
      data.top_cosine,
      "score",
      "Cosine Score",
      "higher is better"
    );

    resultsEl.innerHTML = html;
    statusEl.textContent = "Done.";
  } catch (e) {
    statusEl.textContent = "Backend error. Make sure server.js is running.";
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", doSearch);
