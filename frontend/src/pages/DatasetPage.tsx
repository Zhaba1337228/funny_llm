import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { DataTable } from '../components/DataTable'
import { LoadingState } from '../components/LoadingState'
import { PlotPanel } from '../components/PlotPanel'
import { SectionHeader } from '../components/SectionHeader'
import { client } from '../lib/api'
import { formatMetricLabel, formatNumber } from '../lib/format'
import type { DatasetOverviewResponse, EdaSummaryResponse, PreviewResponse } from '../types/api'

export const DatasetPage = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetOverviewResponse | null>(null)
  const [eda, setEda] = useState<EdaSummaryResponse | null>(null)
  const [preview, setPreview] = useState<PreviewResponse | null>(null)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState<string>('candidate_id')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const loadSummary = async () => {
      const [dataset, edaSummary] = await Promise.all([client.datasetInfo(), client.edaSummary()])
      if (!mounted) return
      setDatasetInfo(dataset)
      setEda(edaSummary)
    }
    void loadSummary()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    let mounted = true
    const loadPreview = async () => {
      setLoading(true)
      const response = await client.datasetPreview({ page, page_size: 15, search, sort_by: sortBy, sort_dir: sortDir })
      if (mounted) {
        setPreview(response)
        setLoading(false)
      }
    }
    void loadPreview()
    return () => {
      mounted = false
    }
  }, [page, search, sortBy, sortDir])

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortDir((value) => (value === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortBy(column)
    setSortDir('asc')
  }

  if (!datasetInfo || !eda) {
    return <LoadingState label="Inspecting dataset structure..." />
  }

  const missingPairs = Object.entries(datasetInfo.dataset.missing_values).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Dataset explorer"
        title="Candidate data profile"
        description="Inspect schema, preview rows, understand missingness, and verify whether the application is using a native target or synthetic scoring mode."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <InfoCard title="Rows" value={formatNumber(datasetInfo.dataset.num_rows)} />
        <InfoCard title="Columns" value={formatNumber(datasetInfo.dataset.num_columns)} />
        <InfoCard title="Numerical features" value={formatNumber(datasetInfo.dataset.numeric_columns.length)} />
        <InfoCard title="Categorical features" value={formatNumber(datasetInfo.dataset.categorical_columns.length)} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PlotPanel
          title="Target / class balance"
          description="Current distribution for the detected target"
          data={[
            {
              type: 'bar',
              x: eda.target_distribution.x,
              y: eda.target_distribution.y,
              marker: { color: '#34d399' },
            },
          ]}
        />
        <PlotPanel
          title="Missing values"
          description="The provided Kaggle dataset is expected to be complete"
          data={[
            {
              type: 'bar',
              x: missingPairs.map(([column]) => formatMetricLabel(column)),
              y: missingPairs.map(([, count]) => count),
              marker: { color: '#60a5fa' },
            },
          ]}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.4fr,1fr]">
        <Card title="Candidate preview" description="Search, sort, and paginate through the raw candidate dataset">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row">
            <input
              value={search}
              onChange={(event) => {
                setPage(1)
                setSearch(event.target.value)
              }}
              placeholder="Search candidate id, education, company type..."
              className="w-full rounded-2xl border border-white/10 bg-slate-950/20 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-cyan-400/40"
            />
          </div>
          {loading || !preview ? (
            <LoadingState label="Loading preview rows..." />
          ) : (
            <div className="space-y-4">
              <DataTable
                columns={preview.columns.slice(0, 9)}
                rows={preview.rows}
                sortBy={sortBy}
                sortDir={sortDir}
                onSort={handleSort}
                compact
              />
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>
                  Page {page} · {formatNumber(preview.total_rows)} filtered rows
                </span>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setPage((value) => Math.max(1, value - 1))}
                    className="rounded-full border border-white/10 px-4 py-2 text-slate-200 transition hover:border-cyan-400/30"
                  >
                    Prev
                  </button>
                  <button
                    type="button"
                    onClick={() => setPage((value) => value + 1)}
                    className="rounded-full border border-white/10 px-4 py-2 text-slate-200 transition hover:border-cyan-400/30"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card title="Feature schema" description="Auto-detected feature types and useful defaults">
          <div className="space-y-3">
            {datasetInfo.feature_schema.map((feature) => (
              <div key={feature.name} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="font-medium text-white">{formatMetricLabel(feature.name)}</div>
                    <div className="mt-1 text-sm text-slate-400">{feature.description}</div>
                  </div>
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-300">
                    {feature.type}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

const InfoCard = ({ title, value }: { title: string; value: string }) => (
  <Card title={title}>
    <div className="font-display text-3xl font-semibold text-white">{value}</div>
  </Card>
)
