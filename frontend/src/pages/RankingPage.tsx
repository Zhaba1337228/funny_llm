import { Search, SlidersHorizontal } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Card } from '../components/Card'
import { DataTable } from '../components/DataTable'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { SectionHeader } from '../components/SectionHeader'
import { client } from '../lib/api'
import { formatNumber } from '../lib/format'
import type { RankingResponse } from '../types/api'

export const RankingPage = () => {
  const navigate = useNavigate()
  const [ranking, setRanking] = useState<RankingResponse | null>(null)
  const [search, setSearch] = useState('')
  const [minScore, setMinScore] = useState(60)
  const [experienceMin, setExperienceMin] = useState(0)
  const [skillMin, setSkillMin] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      setLoading(true)
      const response = await client.topCandidates({
        limit: 60,
        search,
        min_score: minScore,
        experience_min: experienceMin,
        skill_score_min: skillMin,
      }).catch(() => null)
      if (mounted) {
        setRanking(response)
        setLoading(false)
      }
    }
    void load()
    return () => {
      mounted = false
    }
  }, [search, minScore, experienceMin, skillMin])

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Candidate ranking"
        title="Top-scoring talent shortlist"
        description="Filter the ranked candidate queue by score, experience, and skill signals, then open any candidate to inspect explainability details."
        action={
          <a
            href={client.exportCandidatesUrl()}
            className="rounded-full border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-100 transition hover:bg-emerald-500/20"
          >
            Export CSV
          </a>
        }
      />

      <Card title="Filters" description="Narrow the ranking to match recruiter intent">
        <div className="grid gap-4 lg:grid-cols-4">
          <label className="space-y-2">
            <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Search</span>
            <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-slate-950/20 px-4">
              <Search className="h-4 w-4 text-slate-500" />
              <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="ID, education, company..." className="w-full bg-transparent py-3 text-sm text-white outline-none placeholder:text-slate-500" />
            </div>
          </label>
          <FilterSlider label="Min score" value={minScore} onChange={setMinScore} max={100} />
          <FilterSlider label="Min experience" value={experienceMin} onChange={setExperienceMin} max={10} step={0.5} />
          <FilterSlider label="Min skills score" value={skillMin} onChange={setSkillMin} max={100} />
        </div>
      </Card>

      {loading ? (
        <LoadingState label="Scoring candidate ranking..." />
      ) : !ranking || ranking.rows.length === 0 ? (
        <EmptyState title="No ranked candidates" description="Train a model first or relax the active filters to see ranked profiles." />
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <SummaryCard title="Matching candidates" value={formatNumber(ranking.total_rows)} />
            <SummaryCard title="Active model" value={ranking.active_model || '-'} />
            <SummaryCard title="Task mode" value={ranking.task_type || '-'} />
          </div>
          <Card title="Ranked candidates" description="Click a row to open the detailed profile">
            <DataTable
              columns={['rank', 'candidate_id', 'predicted_score', 'hire_probability', 'recommendation', 'experience_years', 'skills_score', 'education_level']}
              rows={ranking.rows}
              compact
              onRowClick={(row) => navigate(`/candidate/${row.candidate_id}`)}
            />
          </Card>
        </>
      )}
    </div>
  )
}

const FilterSlider = ({
  label,
  value,
  onChange,
  max,
  step = 1,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  max: number
  step?: number
}) => (
  <label className="space-y-2">
    <span className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-slate-400">
      <SlidersHorizontal className="h-3.5 w-3.5" />
      {label}
    </span>
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
      <div className="mb-2 flex items-center justify-between text-sm text-white">
        <span>{value}</span>
        <span className="text-slate-400">max {max}</span>
      </div>
      <input type="range" min={0} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} className="w-full accent-cyan-300" />
    </div>
  </label>
)

const SummaryCard = ({ title, value }: { title: string; value: string }) => (
  <Card title={title}>
    <div className="font-display text-3xl font-semibold text-white">{value}</div>
  </Card>
)
