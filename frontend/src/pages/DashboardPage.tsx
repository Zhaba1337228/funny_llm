import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { PlotPanel } from '../components/PlotPanel'
import { SectionHeader } from '../components/SectionHeader'
import { StatusBadge } from '../components/StatusBadge'
import { client } from '../lib/api'
import { formatNumber, formatPercent } from '../lib/format'
import type { DatasetOverviewResponse, DeviceInfo, EdaSummaryResponse, RankingResponse, TrainingResults, TrainingStatus } from '../types/api'

export const DashboardPage = ({
  trainingStatus,
  deviceInfo,
}: {
  trainingStatus: TrainingStatus | null
  deviceInfo: DeviceInfo | null
}) => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetOverviewResponse | null>(null)
  const [eda, setEda] = useState<EdaSummaryResponse | null>(null)
  const [results, setResults] = useState<TrainingResults | null>(null)
  const [ranking, setRanking] = useState<RankingResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      try {
        if (!datasetInfo || !eda) {
          setLoading(true)
        }
        setLoadError(null)
        const [datasetResult, edaResult, topCandidatesResult, trainingResult] = await Promise.allSettled([
          client.datasetInfo(),
          client.edaSummary(),
          client.topCandidates({ limit: 8 }),
          client.trainingResults(),
        ])
        if (!mounted) return
        if (datasetResult.status === 'fulfilled') {
          setDatasetInfo(datasetResult.value)
        }
        if (edaResult.status === 'fulfilled') {
          setEda(edaResult.value)
        }
        if (topCandidatesResult.status === 'fulfilled') {
          setRanking(topCandidatesResult.value)
        }
        if (trainingResult.status === 'fulfilled') {
          setResults(trainingResult.value)
        } else {
          setResults(null)
        }

        if (datasetResult.status === 'rejected' && edaResult.status === 'rejected' && !datasetInfo && !eda) {
          setLoadError('Dashboard data is temporarily unavailable. Backend may still be warming up.')
        }
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }
    void load()
    return () => {
      mounted = false
    }
  }, [trainingStatus?.status])

  if (loading) {
    return <LoadingState label="Compiling dashboard overview..." />
  }

  if (loadError && !datasetInfo && !eda) {
    return <EmptyState title="Dashboard warming up" description={loadError} />
  }

  if (!datasetInfo || !eda) {
    return <EmptyState title="Dataset unavailable" description="Backend could not load the resume dataset. Check Kaggle download access and API logs." />
  }

  const targetDistribution = eda.target_distribution
  const topCandidates = ranking?.rows?.slice(0, 8) || []

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Executive overview"
        title="AI recruitment command center"
        description="Monitor dataset health, track model readiness, and highlight the strongest candidates before they ever reach the recruiter queue."
        action={<StatusBadge label={trainingStatus?.status || 'idle'} tone={trainingStatus?.status === 'trained' ? 'emerald' : trainingStatus?.status === 'training' ? 'cyan' : 'slate'} />}
      />

      {trainingStatus?.status === 'training' && (
        <Card
          title="Live training stream"
          description="Watch the active run in real time while metrics, ranking, and artifacts are still being produced."
          className="border-cyan-400/20 bg-cyan-500/10"
        >
          <div className="space-y-4">
            <div className="h-3 overflow-hidden rounded-full bg-slate-950/50">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-sky-300 to-emerald-400 transition-all"
                style={{ width: `${Math.max((trainingStatus.progress || 0) * 100, 6)}%` }}
              />
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <LiveInfo label="Current step" value={trainingStatus.current_step || 'Preparing run'} />
              <LiveInfo label="Active model" value={trainingStatus.active_model_name || 'Initializing'} />
              <LiveInfo label="Elapsed" value={trainingStatus.elapsed_seconds ? `${trainingStatus.elapsed_seconds}s` : '-'} />
              <LiveInfo label="Device" value={trainingStatus.device || deviceInfo?.gpu_name || deviceInfo?.preferred_training_device || 'auto'} />
            </div>

            <div className="max-h-[220px] space-y-2 overflow-auto rounded-[1.25rem] border border-white/10 bg-slate-950/40 p-4 font-mono text-xs text-slate-200">
              {(trainingStatus.logs || []).length > 0 ? (
                trainingStatus.logs.slice(-12).map((log) => <div key={log}>{log}</div>)
              ) : (
                <div>No live logs yet. The backend is still preparing the training job.</div>
              )}
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Candidates" description="Rows currently available for analysis">
          <div className="font-display text-3xl font-semibold text-white">{formatNumber(datasetInfo.overview.candidate_count)}</div>
        </Card>
        <Card title="Features" description="Numeric + categorical inputs used for modeling">
          <div className="font-display text-3xl font-semibold text-white">{formatNumber(datasetInfo.overview.feature_count)}</div>
        </Card>
        <Card title="Active model" description="Current production artifact">
          <div className="font-display text-2xl font-semibold text-white">{results?.model_name || 'Not trained'}</div>
          <div className="mt-3 text-sm text-slate-400">Task: {results?.task_type || datasetInfo.overview.available_tasks.join(' / ')}</div>
        </Card>
        <Card title="Training device" description="Automatic device detection">
          <div className="font-display text-2xl font-semibold text-white">{deviceInfo?.gpu_name || deviceInfo?.preferred_training_device || '-'}</div>
          <div className="mt-3 text-sm text-slate-400">CUDA available: {deviceInfo?.cuda_available ? 'yes' : 'no'}</div>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.4fr,1fr]">
        <Card title="Model health" description="The most recent validation snapshot">
          {results ? (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {results.task_type === 'classification' ? (
                <>
                  <MetricTile label="Accuracy" value={formatPercent(results.metrics.accuracy, 1)} />
                  <MetricTile label="F1" value={formatNumber(results.metrics.f1, 3)} />
                  <MetricTile label="ROC-AUC" value={formatNumber(results.metrics.roc_auc, 3)} />
                  <MetricTile label="Precision" value={formatNumber(results.metrics.precision, 3)} />
                </>
              ) : (
                <>
                  <MetricTile label="MAE" value={formatNumber(results.metrics.mae, 3)} />
                  <MetricTile label="RMSE" value={formatNumber(results.metrics.rmse, 3)} />
                  <MetricTile label="R2" value={formatNumber(results.metrics.r2, 3)} />
                  <MetricTile label="Target" value={results.target_column} />
                </>
              )}
            </div>
          ) : (
            <EmptyState title="No trained model yet" description="Open the Training Lab to fit a classifier or score model and unlock performance analytics." />
          )}
        </Card>

        <Card title="Dataset mode" description="How the platform interprets supervision">
          <div className="space-y-3 text-sm text-slate-300">
            <div>Classification target: {datasetInfo.dataset.target_columns.classification || 'Unavailable'}</div>
            <div>Regression target: {datasetInfo.dataset.target_columns.regression}</div>
            <div>Synthetic target active: {datasetInfo.dataset.synthetic_target_active ? 'yes' : 'no'}</div>
            <div className="rounded-2xl border border-cyan-400/15 bg-cyan-400/10 p-4 text-slate-200">
              {datasetInfo.dataset.synthetic_target_logic}
            </div>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PlotPanel
          title="Target distribution"
          description="Class balance and scoring target availability"
          data={[
            {
              type: 'bar',
              x: targetDistribution.x,
              y: targetDistribution.y,
              marker: { color: ['#2dd4bf', '#38bdf8', '#f59e0b', '#f97316'] },
            },
          ]}
        />
        <PlotPanel
          title="Correlation heatmap"
          description="Numerical relationships across the resume dataset"
          data={[
            {
              type: 'heatmap',
              z: eda.correlation.matrix,
              x: eda.correlation.labels,
              y: eda.correlation.labels,
              colorscale: [
                [0, '#0f172a'],
                [0.5, '#0ea5e9'],
                [1, '#34d399'],
              ],
            },
          ]}
          layout={{ height: 360 }}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr,1fr]">
        <PlotPanel
          title="Top ranked candidates"
          description="Highest-scoring profiles under the active model"
          data={[
            {
              type: 'bar',
              x: topCandidates.map((row) => String(row.candidate_id)),
              y: topCandidates.map((row) => Number(row.predicted_score || 0)),
              marker: { color: '#22d3ee' },
            },
          ]}
        />

        <Card title="Recruiter notes" description="Fast context for the current screening setup">
          <ul className="space-y-4 text-sm text-slate-300">
            <li>Dataset path: <span className="text-slate-100">{datasetInfo.dataset.dataset_name}</span></li>
            <li>Candidate hire rate: <span className="text-slate-100">{formatPercent(Number(datasetInfo.dataset.target_distribution['1'] ?? 0), 1)}</span></li>
            <li>Training step: <span className="text-slate-100">{trainingStatus?.current_step || 'Idle'}</span></li>
            <li>Available tasks: <span className="text-slate-100">{datasetInfo.overview.available_tasks.join(', ')}</span></li>
          </ul>
        </Card>
      </div>
    </div>
  )
}

const MetricTile = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
    <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
    <div className="mt-3 font-display text-2xl font-semibold text-white">{value}</div>
  </div>
)

const LiveInfo = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
    <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
    <div className="mt-2 text-sm text-white">{value}</div>
  </div>
)
