export type TaskType = 'classification' | 'regression'
export type TrainingProfile = 'rapid' | 'balanced' | 'max_accuracy' | 'server_max'

export interface DatasetOverviewResponse {
  dataset: {
    dataset_path: string
    dataset_name: string
    num_rows: number
    num_columns: number
    numeric_columns: string[]
    categorical_columns: string[]
    feature_types: Record<string, string>
    missing_values: Record<string, number>
    target_distribution: Record<string, number>
    summary_statistics: Record<string, Record<string, number | string | null>>
    target_columns: {
      classification: string | null
      regression: string
    }
    synthetic_target_active: boolean
    synthetic_target_logic: string
  }
  feature_schema: FeatureSchema[]
  overview: {
    candidate_count: number
    feature_count: number
    available_tasks: TaskType[]
    classification_target: string | null
    regression_target: string
    numeric_summary: Record<string, Record<string, number>>
  }
}

export interface FeatureSchema {
  name: string
  type: 'numeric' | 'categorical'
  description: string
  min?: number
  max?: number
  mean?: number
  median?: number
  categories?: Array<{ label: string; count: number }>
}

export interface PreviewResponse {
  total_rows: number
  page: number
  page_size: number
  rows: Record<string, unknown>[]
  columns: string[]
}

export interface EdaSummaryResponse {
  missing_values: Record<string, number>
  target_distribution: { x: string[]; y: number[] }
  class_balance: Record<string, number>
  numeric_distributions: Record<string, { x: number[]; y: number[] }>
  box_plots: Record<string, { min: number; q1: number; median: number; q3: number; max: number }>
  categorical_distributions: Record<string, { x: string[]; y: number[] }>
  correlation: { labels: string[]; matrix: number[][] }
  dataset_summary: {
    row_count: number
    column_count: number
    numeric_feature_count: number
    categorical_feature_count: number
  }
  sample_rows: Record<string, unknown>[]
}

export interface TrainingStatus {
  run_id: string | null
  status: string
  progress: number
  current_step: string | null
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number | null
  logs: string[]
  device: string | null
  active_model_name: string | null
  task_type: TaskType | null
  can_stop: boolean
}

export interface TrainingResults {
  run_id: string
  trained_at: string
  task_type: TaskType
  training_profile?: TrainingProfile
  model_name: string
  target_column: string
  feature_columns: string[]
  metrics: Record<string, any>
  history: Record<string, number[]>
  comparison: ModelComparisonEntry[]
  feature_importance: Array<{ feature: string; importance: number }>
  synthetic_mode: boolean
  device: string
}

export interface ModelComparisonEntry {
  model_name: string
  label: string
  kind: string
  training_time_seconds: number
  device: string
  metrics: Record<string, any>
  score_for_selection: number
  score_direction: string
}

export interface ModelListResponse {
  catalog: Array<{
    name: string
    label: string
    task_types: TaskType[]
    kind: string
    description: string
    default_params: Record<string, unknown>
  }>
  training_profiles: Array<{
    name: TrainingProfile
    label: string
    description: string
  }>
  saved_versions: Array<{
    run_id: string
    created_at: string
    model_name: string
    task_type: TaskType
    target_column: string
    metrics: Record<string, unknown>
    is_active: boolean
    resumable?: boolean
  }>
  recent_experiments: Array<{
    run_id: string
    trained_at: string
    task_type: TaskType
    model_name: string
    target_column: string
    metrics: Record<string, unknown>
    device: string
  }>
}

export interface DeviceInfo {
  python_version: string
  platform: string
  processor: string
  cpu_count: number
  torch_available: boolean
  cuda_available: boolean
  gpu_name: string | null
  gpu_count: number
  gpu_devices: Array<{
    index: number
    name: string
    memory_gb: number
    multi_processor_count: number
  }>
  total_gpu_memory_gb: number
  preferred_training_device: string
}

export interface RankingResponse {
  total_rows: number
  rows: Record<string, any>[]
  active_model: string | null
  task_type: TaskType | null
}

export interface CandidateDetailResponse {
  candidate: Record<string, any>
  explanation: string
  strengths: string[]
  weaknesses: string[]
  feature_contributions: Array<{
    feature: string
    label: string
    impact: number
    importance: number
    value: string | number
  }>
}

export interface PredictionResponse {
  candidate_score: number
  hire_probability: number
  recommendation: string
  explanation: string
  strengths: string[]
  weaknesses: string[]
  feature_contributions: Array<{
    feature: string
    label: string
    impact: number
    importance: number
    value: string | number
  }>
}
