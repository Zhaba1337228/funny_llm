import { Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { PlotPanel } from '../components/PlotPanel'
import { SectionHeader } from '../components/SectionHeader'
import { StatusBadge } from '../components/StatusBadge'
import { client, getApiErrorMessage } from '../lib/api'
import { formatMetricLabel, formatNumber, recommendationTone } from '../lib/format'
import type { DatasetOverviewResponse, FeatureSchema, PredictionResponse } from '../types/api'

export const PlaygroundPage = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetOverviewResponse | null>(null)
  const [formValues, setFormValues] = useState<Record<string, string | number>>({})
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      const response = await client.datasetInfo().catch(() => null)
      if (!mounted || !response) return
      const defaults = response.feature_schema.reduce<Record<string, string | number>>((accumulator, feature) => {
        accumulator[feature.name] =
          feature.type === 'numeric' ? feature.median ?? feature.mean ?? 0 : feature.categories?.[0]?.label ?? ''
        return accumulator
      }, {})
      setDatasetInfo(response)
      setFormValues(defaults)
      setLoading(false)
    }
    void load()
    return () => {
      mounted = false
    }
  }, [])

  const submitPrediction = async () => {
    try {
      setErrorMessage(null)
      const response = await client.predict(formValues)
      setPrediction(response)
    } catch (error) {
      setErrorMessage(getApiErrorMessage(error))
    }
  }

  if (loading) return <LoadingState label="Preparing interactive prediction panel..." />
  if (!datasetInfo) {
    return <EmptyState title="Playground unavailable" description="Feature schema could not be loaded from the dataset service." />
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Prediction playground"
        title="Interactive AI evaluation panel"
        description="Tune the candidate profile manually and let the active model return a score, hire probability, recommendation, and explanation in real time."
        action={
          <button
            type="button"
            onClick={() => void submitPrediction()}
            className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-500/20"
          >
            <Sparkles className="h-4 w-4" />
            Evaluate candidate
          </button>
        }
      />

      {errorMessage && (
        <Card className="border-rose-400/20 bg-rose-500/10">
          <div className="text-sm text-rose-100">{errorMessage}</div>
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-[1.1fr,0.9fr]">
        <Card title="Candidate inputs" description="The form is generated directly from the discovered dataset schema">
          <div className="grid gap-4 md:grid-cols-2">
            {datasetInfo.feature_schema.map((feature) => (
              <FeatureField
                key={feature.name}
                feature={feature}
                value={formValues[feature.name]}
                onChange={(value) =>
                  setFormValues((current) => ({
                    ...current,
                    [feature.name]: value,
                  }))
                }
              />
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          {prediction ? (
            <>
              <Card title="Prediction result" description="Score + recommendation from the active model">
                <div className="space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="font-display text-4xl font-semibold text-white">{formatNumber(prediction.candidate_score, 2)}</div>
                      <div className="mt-2 text-sm text-slate-400">Candidate score</div>
                    </div>
                    <StatusBadge label={prediction.recommendation} tone={recommendationTone(prediction.recommendation)} />
                  </div>
                  <div className="rounded-[1.5rem] border border-white/10 bg-white/5 p-4">
                    <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Hire probability</div>
                    <div className="mt-2 font-display text-2xl font-semibold text-white">{formatNumber(prediction.hire_probability * 100, 1)}%</div>
                  </div>
                  <div className="rounded-[1.5rem] border border-cyan-400/20 bg-cyan-500/10 p-4 text-sm text-slate-100">
                    {prediction.explanation}
                  </div>
                </div>
              </Card>
              <PlotPanel
                title="Top contributing factors"
                description="Why the model responded this way"
                data={[
                  {
                    type: 'bar',
                    orientation: 'h',
                    y: prediction.feature_contributions.map((item) => item.label).reverse(),
                    x: prediction.feature_contributions.map((item) => item.impact).reverse(),
                    marker: {
                      color: prediction.feature_contributions.map((item) => (item.impact >= 0 ? '#34d399' : '#fb923c')).reverse(),
                    },
                  },
                ]}
                layout={{ height: 380, margin: { l: 130, r: 12, t: 12, b: 30 } }}
              />
            </>
          ) : (
            <EmptyState title="No prediction yet" description="Complete the form and run the evaluator to generate a candidate score, recommendation, and explanation." />
          )}
        </div>
      </div>
    </div>
  )
}

const FeatureField = ({
  feature,
  value,
  onChange,
}: {
  feature: FeatureSchema
  value: string | number | undefined
  onChange: (value: string | number) => void
}) => (
  <label className="space-y-2">
    <span className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">{formatMetricLabel(feature.name)}</span>
    {feature.type === 'numeric' ? (
      <input
        type="number"
        value={value ?? ''}
        min={feature.min}
        max={feature.max}
        step={feature.name.includes('score') || feature.name.includes('cgpa') || feature.name.includes('years') ? 0.1 : 1}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full rounded-2xl border border-white/10 bg-slate-950/20 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40"
      />
    ) : (
      <select
        value={String(value ?? '')}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-2xl border border-white/10 bg-slate-950/20 px-4 py-3 text-sm text-white outline-none transition focus:border-cyan-400/40"
      >
        {feature.categories?.map((category) => (
          <option key={category.label} value={category.label}>
            {category.label}
          </option>
        ))}
      </select>
    )}
  </label>
)
