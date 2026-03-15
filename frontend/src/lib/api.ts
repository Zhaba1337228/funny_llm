import axios from 'axios'

import type {
  CandidateDetailResponse,
  DatasetOverviewResponse,
  DeviceInfo,
  EdaSummaryResponse,
  ModelListResponse,
  PredictionResponse,
  PreviewResponse,
  RankingResponse,
  TrainingResults,
  TrainingStatus,
} from '../types/api'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 120000,
})

export const client = {
  datasetInfo: async () => (await api.get<DatasetOverviewResponse>('/api/dataset/info')).data,
  datasetPreview: async (params: Record<string, unknown>) =>
    (await api.get<PreviewResponse>('/api/dataset/preview', { params })).data,
  edaSummary: async () => (await api.get<EdaSummaryResponse>('/api/eda/summary')).data,
  trainingStatus: async () => (await api.get<TrainingStatus>('/api/train/status')).data,
  trainingResults: async () => (await api.get<TrainingResults>('/api/train/results')).data,
  startTraining: async (payload: Record<string, unknown>) => (await api.post<TrainingStatus>('/api/train/start', payload)).data,
  stopTraining: async () => (await api.post<TrainingStatus>('/api/train/stop')).data,
  modelsList: async () => (await api.get<ModelListResponse>('/api/models/list')).data,
  modelsCompare: async () => (await api.get('/api/models/compare')).data,
  selectModel: async (runId: string) => (await api.post(`/api/models/select/${runId}`)).data,
  predict: async (features: Record<string, unknown>) =>
    (await api.post<PredictionResponse>('/api/predict', { features })).data,
  topCandidates: async (params: Record<string, unknown>) =>
    (await api.get<RankingResponse>('/api/candidates/top', { params })).data,
  candidateDetail: async (candidateId: string | number) =>
    (await api.get<CandidateDetailResponse>(`/api/candidates/${candidateId}`)).data,
  deviceInfo: async () => (await api.get<DeviceInfo>('/api/system/device')).data,
  exportCandidatesUrl: () => `${import.meta.env.VITE_API_BASE_URL || ''}/api/candidates/export`,
}

export const getApiErrorMessage = (error: unknown) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail
    if (typeof detail === 'string') {
      return detail
    }
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unexpected API error.'
}

export default api
