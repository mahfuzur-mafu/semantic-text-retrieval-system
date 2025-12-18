import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { pipeline } from "@xenova/transformers";
import { fromArrayBuffer } from "numpy-parser";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUTPUT_DIR = path.join(__dirname, "semantic_gooaq_minilm");
const MODEL_DIR = path.join(OUTPUT_DIR, "model");
const EMB_PATH = path.join(OUTPUT_DIR, "embeddings.npy");
const CHUNKS_PATH = path.join(OUTPUT_DIR, "chunks.json");

const WEB_DIR = path.join(__dirname, "web");
const PORT = 8000;

function l2norm(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.sqrt(s);
}

function euclideanDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function cosineSimilarity(a, b, normA, normB) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot / ((normA * normB) + 1e-10);
}

function getRow(flat, rowIndex, dim) {
  const start = rowIndex * dim;
  return flat.subarray(start, start + dim);
}

async function start() {
  const chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, "utf-8"));

  const ab = fs.readFileSync(EMB_PATH).buffer;
  const np = fromArrayBuffer(ab);

  const N = np.shape[0];
  const D = np.shape[1];

  const embeddingsFlat = new Float32Array(
    np.data.buffer,
    np.data.byteOffset,
    np.data.length
  );

  if (chunks.length !== N) {
    throw new Error(`Mismatch: chunks=${chunks.length} but embeddings rows=${N}`);
  }

  console.log(`Loaded embeddings ${N} x ${D}`);
  console.log("Loading embedding model in Node...");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
    { quantized: false }
  );
  

  console.log("Model loaded.");

  const app = express();
  app.use(express.json({ limit: "1mb" }));
  app.use(express.static(WEB_DIR));

  app.post("/search", async (req, res) => {
    try {
      const prompt = String(req.body?.prompt ?? "").trim();
      let top_k = Number(req.body?.top_k ?? 5);

      if (!prompt) return res.status(400).json({ error: "Empty prompt" });
      if (!Number.isFinite(top_k)) top_k = 5;
      top_k = Math.max(1, Math.min(top_k, 30));

      const out = await extractor(prompt, { pooling: "mean", normalize: false });
      const promptVec = Float32Array.from(out.data);

      if (promptVec.length !== D) {
        return res.status(500).json({
          error: `Prompt dim ${promptVec.length} does not match embeddings dim ${D}`
        });
      }

      const promptNorm = l2norm(promptVec);

      const eucScores = new Float32Array(N);
      const cosScores = new Float32Array(N);

      let bestEucIdx = 0;
      let bestEuc = Number.POSITIVE_INFINITY;

      let bestCosIdx = 0;
      let bestCos = Number.NEGATIVE_INFINITY;

      for (let i = 0; i < N; i++) {
        const row = getRow(embeddingsFlat, i, D);

        const euc = euclideanDistance(row, promptVec);
        eucScores[i] = euc;
        if (euc < bestEuc) {
          bestEuc = euc;
          bestEucIdx = i;
        }

        const cos = cosineSimilarity(row, promptVec, l2norm(row), promptNorm);
        cosScores[i] = cos;
        if (cos > bestCos) {
          bestCos = cos;
          bestCosIdx = i;
        }
      }

      const idxAll = Array.from({ length: N }, (_, i) => i);

      const topEucIdx = idxAll
        .slice()
        .sort((a, b) => eucScores[a] - eucScores[b])
        .slice(0, top_k);

      const topCosIdx = idxAll
        .slice()
        .sort((a, b) => cosScores[b] - cosScores[a])
        .slice(0, top_k);

      res.json({
        embedding_dim: D,
        prompt,
        closest_euclidean: {
          index: bestEucIdx,
          distance: Number(bestEuc),
          text: chunks[bestEucIdx]
        },
        closest_cosine: {
          index: bestCosIdx,
          score: Number(bestCos),
          text: chunks[bestCosIdx]
        },
        top_euclidean: topEucIdx.map((i, r) => ({
          rank: r + 1,
          index: i,
          distance: Number(eucScores[i]),
          text: chunks[i]
        })),
        top_cosine: topCosIdx.map((i, r) => ({
          rank: r + 1,
          index: i,
          score: Number(cosScores[i]),
          text: chunks[i]
        }))
      });
    } catch (e) {
      res.status(500).json({ error: String(e?.message ?? e) });
    }
  });

  app.listen(PORT, () => {
    console.log(`Open http://localhost:${PORT}`);
  });
}

start().catch((e) => {
  console.error(e);
  process.exit(1);s
});
