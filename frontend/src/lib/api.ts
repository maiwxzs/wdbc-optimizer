import type {
  CompareResult,
  DatasetSummary,
  FloatAudit,
  GradientCheck,
  InitMode,
  ModelInfo,
  PCAPoints,
  Prediction,
  StepSweep,
  TrainConfig,
  TrainResult,
} from "./types";

// En desarrollo Vite redirige /api → http://localhost:8000 (ver vite.config.ts).
const BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* respuesta sin JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  summary: () => request<DatasetSummary>("/api/dataset/summary"),
  pca: () => request<PCAPoints>("/api/dataset/pca"),
  model: () => request<ModelInfo>("/api/model"),
  train: (cfg: Partial<TrainConfig>) => request<TrainResult>("/api/train", cfg),
  compare: (max_iter = 3000) => request<CompareResult>("/api/train/compare", { max_iter }),
  predictSample: (sample_index: number) => request<Prediction>("/api/predict", { sample_index }),
  predictFeatures: (features: number[]) => request<Prediction>("/api/predict", { features }),
  gradientCheck: (p: { point: InitMode; h: number; seed: number }) =>
    request<GradientCheck>("/api/audit/gradient-check", p),
  stepSweep: (p: { point: InitMode; exp_min: number; exp_max: number; n: number; seed: number }) =>
    request<StepSweep>("/api/audit/step-sweep", p),
  floatAudit: () => request<FloatAudit>("/api/audit/float"),
};
