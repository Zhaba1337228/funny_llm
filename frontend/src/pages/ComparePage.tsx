import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { DataTable } from '../components/DataTable'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { SectionHeader } from '../components/SectionHeader'
import { client } from '../lib/api'
import { formatDateTime } from '../lib/format'
import type { ModelListResponse, TrainingResults } from '../types/api'

export const ComparePage = () => {
  const [models, setModels] = useState<ModelListResponse | null>(null)
  const [results, setResults] = useState<TrainingResults | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      const [modelList, trainingResults] = await Promise.all([
        client.modelsList().catch(() => null),
        client.trainingResults().catch(() => null),
      ])
      if (mounted) {
        setModels(modelList)
        setResults(trainingResults)
        setLoading(false)
      }
    }
    void load()
    return () => {
      mounted = false
    }
  }, [])

  if (loading) return <LoadingState label="Loading model comparison workspace..." />
  if (!models) {
    return <EmptyState title="Model registry unavailable" description="The backend could not load saved runs or the comparison catalog." />
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Compare models"
        title="Benchmark candidates against multiple learners"
        description="Review the latest experiment table, browse historical model versions, and switch the active production artifact without retraining."
      />

      <div className="grid gap-4 xl:grid-cols-[1.1fr,0.9fr]">
        <Card title="Latest comparison" description="Models included in the most recent experiment">
          {results && results.comparison.length > 0 ? (
            <DataTable columns={['label', 'kind', 'training_time_seconds', 'device', 'score_for_selection']} rows={results.comparison} compact />
          ) : (
            <EmptyState title="No recent comparison" description="Launch the Training Lab with multiple compare models enabled to populate this leaderboard." />
          )}
        </Card>

        <Card title="Available models" description="What the current backend build can train">
          <div className="space-y-3">
            {models.catalog.map((model) => (
              <div key={model.name} className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="font-medium text-white">{model.label}</div>
                    <div className="mt-1 text-sm text-slate-400">{model.description}</div>
                  </div>
                  <div className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-300">{model.kind}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Saved versions" description="Persisted experiments that can be promoted as the active model">
        {models.saved_versions.length === 0 ? (
          <EmptyState title="No saved versions" description="Run training once to start building a reusable model history." />
        ) : (
          <div className="space-y-3">
            {models.saved_versions.map((version) => (
              <div key={version.run_id} className="flex flex-col gap-4 rounded-[1.5rem] border border-white/10 bg-white/5 p-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <div className="font-display text-lg font-semibold text-white">{version.model_name}</div>
                  <div className="mt-1 text-sm text-slate-400">
                    {version.task_type} · target {version.target_column} · {formatDateTime(version.created_at)}
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <div className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-300">
                    {version.is_active ? 'active' : 'saved'}
                  </div>
                  <button
                    type="button"
                    onClick={() => void client.selectModel(version.run_id)}
                    className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-500/20"
                  >
                    Make active
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title="Recent experiments" description="Lightweight training history viewer">
        {models.recent_experiments.length === 0 ? (
          <EmptyState title="No experiment history" description="Recent training runs will appear here once you begin benchmarking models." />
        ) : (
          <DataTable columns={['run_id', 'model_name', 'task_type', 'target_column', 'device']} rows={models.recent_experiments} compact />
        )}
      </Card>
    </div>
  )
}
