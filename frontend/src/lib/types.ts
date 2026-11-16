// Tipos de las respuestas del backend FastAPI. Los valores no finitos (NaN/Inf)
// llegan como null porque JSON no los admite.
export type Num = number | null;
export type StopReason = "grad_norm" | "loss_change" | "max_iter" | "divergence";
export type NumericMode = "logits" | "clipped" | "naive";
export type InitMode = "zeros" | "random" | "trained";

export interface FeatureStat {
  index: number;
  name: string;
  name_es: string;
  group: "mean" | "error" | "worst";
  min: number;
  max: number;
  mean: number;
  std: number;
  mean_benign: number;
  mean_malignant: number;
}

export interface DatasetSummary {
  n_samples: number;
  n_train: number;
  n_test: number;
  n_features: number;
  class_counts: { malignant: number; benign: number };
  label_convention: string;
  pca_explained_variance: number[];
  features: FeatureStat[];
}

export interface Confusion { tp: number; tn: number; fp: number; fn: number }

export interface ClassificationReport {
  confusion: Confusion;
  accuracy: number;
  precision: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  auc: Num;
  threshold: number;
}

export interface History {
  iteration: number[];
  loss: Num[];
  grad_norm: Num[];
  loss_change: Num[];
  weight_norm: Num[];
  recorded_points: number;
}

export interface NumericFlags {
  has_nan: boolean;
  has_inf: boolean;
  exp_overflow_events: number;
  loss_evaluations: number;
  gradient_evaluations: number;
}

export interface WeightEntry { name: string; index: number; weight: Num }

export interface TrainConfig {
  alpha: number;
  max_iter: number;
  tol_grad: number;
  tol_loss: number;
  l2: number;
  mode: NumericMode;
  init: InitMode;
  seed?: number;
  set_active?: boolean;
  max_points?: number;
}

export interface TrainSummary {
  config: TrainConfig & { divergence_threshold: number; record_every: number };
  stop_reason: StopReason;
  message: string;
  converged: boolean;
  diverged: boolean;
  divergence_detail: string | null;
  iterations: number;
  elapsed_ms: number;
  final_loss: Num;
  final_grad_norm: Num;
  numeric_flags: NumericFlags;
  lipschitz_L: number;
  alpha_critical: number;
  alpha_over_critical: number;
  bias: Num;
}

export interface TrainResult extends TrainSummary {
  history: History;
  metrics: { train: ClassificationReport; test: ClassificationReport };
  roc_test: { fpr: number[]; tpr: number[]; thresholds: number[] };
  weights: WeightEntry[];
  active_updated: boolean;
}

export interface CompareRun {
  label: string;
  alpha: number;
  mode: NumericMode;
  l2: number;
  stop_reason: StopReason;
  message: string;
  divergence_detail: string | null;
  iterations: number;
  final_loss: Num;
  final_grad_norm: Num;
  test_accuracy: number;
  alpha_over_critical: number;
  numeric_flags: NumericFlags;
  history: History;
}

export interface CompareResult {
  lipschitz_L: number;
  alpha_critical: number;
  max_iter: number;
  runs: CompareRun[];
}

export interface ModelInfo {
  summary: TrainSummary;
  metrics: { train: ClassificationReport; test: ClassificationReport };
  weights: WeightEntry[];
  bias: number;
}

export interface Contribution {
  name: string;
  index: number;
  raw_value: number;
  std_value: number;
  weight: number;
  contribution: number;
}

export interface Prediction {
  probability_malignant: number;
  logit: number;
  prediction: 0 | 1;
  prediction_label: "Maligno" | "Benigno";
  risk_band: { key: string; label: string };
  true_label: 0 | 1 | null;
  bias: number;
  pca: { x: number; y: number };
  raw_features: number[];
  contributions: Contribution[];
}

export interface PCAPoints {
  train: { x: number; y: number; label: 0 | 1 }[];
  test: { i: number; x: number; y: number; label: 0 | 1; proba: number; pred: 0 | 1 }[];
  explained_variance: number[];
}

export interface GradientCheck {
  point: InitMode;
  h: number;
  loss_at_point: number;
  relative_error: Num;
  max_abs_error: Num;
  analytic_norm: number;
  numeric_norm: number;
  verdict: "excelente" | "aceptable" | "sospechoso" | "incorrecto" | "fallo";
  message: string;
  loss_evaluations: number;
  components: { name: string; analytic: number; numeric: number; abs_error: number; rel_error: number }[];
}

export interface StepSweep {
  h: number[];
  central_error: Num[];
  forward_error: Num[];
  central_model: Num[];
  forward_model: Num[];
  central_best: { h: number; error: number };
  forward_best: { h: number; error: number };
  central_h_opt_fit: Num;
  forward_h_opt_fit: Num;
  central_h_opt_theory: number;
  forward_h_opt_theory: number;
  eps_machine: number;
  verdict_at_best: string;
  point: InitMode;
  loss_at_point: number;
}

export type IeeeValue = number | "NaN" | "+inf" | "-inf";

export interface FloatAudit {
  z: number[];
  sigmoid_naive: Num[];
  sigmoid_stable: Num[];
  loss_y1: { exact: Num[]; naive: Num[]; clipped: Num[] };
  loss_y0: { exact: Num[]; naive: Num[] };
  table: {
    z: number;
    sigmoid_naive: IeeeValue;
    sigmoid_stable: IeeeValue;
    exp_overflow: boolean;
    loss_y1_naive: IeeeValue;
    loss_y1_clip: IeeeValue;
    loss_y1_exact: IeeeValue;
    loss_y0_naive: IeeeValue;
    loss_y0_exact: IeeeValue;
  }[];
  constants: Record<string, number>;
  notes: string[];
}
