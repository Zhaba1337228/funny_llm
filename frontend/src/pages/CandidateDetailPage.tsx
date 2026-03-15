import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

import { Card } from '../components/Card'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { PlotPanel } from '../components/PlotPanel'
import { SectionHeader } from '../components/SectionHeader'
import { StatusBadge } from '../components/StatusBadge'
import { client } from '../lib/api'
import { formatMetricLabel, formatNumber, recommendationTone } from '../lib/format'
import type { CandidateDetailResponse } from '../types/api'

export const CandidateDetailPage = () => {
  const { candidateId } = useParams()
  const [detail, setDetail] = useState<CandidateDetailResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      if (!candidateId) return
      const response = await client.candidateDetail(candidateId).catch(() => null)
      if (mounted) {
        setDetail(response)
        setLoading(false)
      }
    }
    void load()
    return () => {
      mounted = false
    }
  }, [candidateId])

  if (loading) return <LoadingState label="Loading candidate detail..." />
  if (!detail) {
    return <EmptyState title="Candidate not available" description="The selected candidate was not found in the ranked cache. Re-run scoring or choose another profile." />
  }

  const candidate = detail.candidate

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Candidate detail"
        title={`Candidate #${candidate.candidate_id}`}
        description={detail.explanation}
        action={<StatusBadge label={String(candidate.recommendation)} tone={recommendationTone(String(candidate.recommendation))} />}
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Predicted score">
          <div className="font-display text-3xl font-semibold text-white">{formatNumber(Number(candidate.predicted_score), 2)}</div>
        </Card>
        <Card title="Hire probability">
          <div className="font-display text-3xl font-semibold text-white">{formatNumber(Number(candidate.hire_probability) * 100, 1)}%</div>
        </Card>
        <Card title="Rank">
          <div className="font-display text-3xl font-semibold text-white">{formatNumber(Number(candidate.rank))}</div>
        </Card>
        <Card title="Recommendation">
          <div className="font-display text-2xl font-semibold text-white">{String(candidate.recommendation)}</div>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr,0.9fr]">
        <Card title="Explanation" description="Human-readable screening rationale">
          <div className="space-y-5">
            <div className="rounded-[1.5rem] border border-cyan-400/20 bg-cyan-500/10 p-4 text-sm text-slate-100">{detail.explanation}</div>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <div className="mb-3 text-xs uppercase tracking-[0.2em] text-emerald-300">Strengths</div>
                <div className="space-y-2">
                  {detail.strengths.map((item) => (
                    <div key={item} className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
                      {item}
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-3 text-xs uppercase tracking-[0.2em] text-amber-300">Weaknesses</div>
                <div className="space-y-2">
                  {detail.weaknesses.map((item) => (
                    <div key={item} className="rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>

        <PlotPanel
          title="Feature contributions"
          description="Approximate directional impact on the final prediction"
          data={[
            {
              type: 'bar',
              orientation: 'h',
              y: detail.feature_contributions.map((item) => item.label).reverse(),
              x: detail.feature_contributions.map((item) => item.impact).reverse(),
              marker: {
                color: detail.feature_contributions.map((item) => (item.impact >= 0 ? '#34d399' : '#f59e0b')).reverse(),
              },
            },
          ]}
          layout={{ height: 420, margin: { l: 130, r: 16, t: 16, b: 30 } }}
        />
      </div>

      <Card title="Candidate profile" description="Raw features and derived ranking outputs">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Object.entries(candidate).map(([key, value]) => (
            <div key={key} className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{formatMetricLabel(key)}</div>
              <div className="mt-2 text-sm text-white">{String(value)}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
