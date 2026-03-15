import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { PlotPanel } from '../components/PlotPanel'
import { SectionHeader } from '../components/SectionHeader'
import { DataTable } from '../components/DataTable'
import { client } from '../lib/api'
import { formatMetricLabel, formatNumber } from '../lib/format'
import type { TrainingResults } from '../types/api'

export const PerformancePage = () => {
  const [results, setResults] = useState<TrainingResults | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      const response = await client.trainingResults().catch(() => null)
      if (mounted) {
        setResults(response)
        setLoading(false)
      }
    }
    void load()
    return () => {
      mounted = false
    }
  }, [])

  if (loading) return <LoadingState label="Loading model performance..." />
  if (!results) {
    return <EmptyState title="No performance report yet" description="Run training first to unlock confusion matrix, ROC curve, feature importance, and neural history plots." />
  }

  const featureImportance = results.feature_importance.slice(0, 10)

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Model performance"
        title="Validation and holdout analytics"
        description="Review accuracy metrics, inspect feature influence, and compare experiment curves before promoting a model."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {Object.entries(results.task_type === 'classification'
          ? {
              accuracy: results.metrics.accuracy,
              precision: results.metrics.precision,
              recall: results.metrics.recall,
              f1: results.metrics.f1,
            }
          : {
              mae: results.metrics.mae,
              rmse: results.metrics.rmse,
              r2: results.metrics.r2,
              target: results.target_column,
            }).map(([key, value]) => (
          <Card key={key} title={formatMetricLabel(key)}>
            <div className="font-display text-3xl font-semibold text-white">
              {typeof value === 'number' ? formatNumber(value, 3) : String(value)}
            </div>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <PlotPanel
          title="Feature importance"
          description="Permutation-based global influence on the active model"
          data={[
            {
              type: 'bar',
              orientation: 'h',
              y: featureImportance.map((item) => formatMetricLabel(item.feature)).reverse(),
              x: featureImportance.map((item) => item.importance).reverse(),
              marker: { color: '#22d3ee' },
            },
          ]}
          layout={{ height: 380, margin: { l: 140, r: 10, t: 10, b: 30 } }}
        />

        {results.task_type === 'classification' ? (
          <PlotPanel
            title="ROC curve"
            description="Threshold trade-offs on the holdout set"
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: results.metrics.roc_curve?.fpr || [],
                y: results.metrics.roc_curve?.tpr || [],
                line: { color: '#34d399', width: 3 },
              },
            ]}
            layout={{ xaxis: { title: { text: 'False positive rate' } }, yaxis: { title: { text: 'True positive rate' } } }}
          />
        ) : (
          <PlotPanel
            title="Prediction error"
            description="True vs predicted candidate scores"
            data={[
              {
                type: 'scatter',
                mode: 'markers',
                x: results.metrics.prediction_error?.y_true || [],
                y: results.metrics.prediction_error?.y_pred || [],
                marker: { color: '#38bdf8', size: 8, opacity: 0.75 },
              },
            ]}
            layout={{ xaxis: { title: { text: 'True score' } }, yaxis: { title: { text: 'Predicted score' } } }}
          />
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr,0.8fr]">
        <Card title="Training curves" description="Visible when the PyTorch neural net was part of the latest run">
          {Object.keys(results.history).length > 0 ? (
            <PlotPanel
              title="Loss history"
              data={[
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: results.history.train_loss?.map((_, index) => index + 1) || [],
                  y: results.history.train_loss || [],
                  name: 'Train loss',
                  line: { color: '#38bdf8' },
                },
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: results.history.val_loss?.map((_, index) => index + 1) || [],
                  y: results.history.val_loss || [],
                  name: 'Validation loss',
                  line: { color: '#34d399' },
                },
              ]}
            />
          ) : (
            <EmptyState title="No neural history recorded" description="Classical models save metrics and feature importance, but not per-epoch curves." />
          )}
        </Card>

        <Card title="Confusion matrix / report" description="Classification report or regression summary">
          {results.task_type === 'classification' ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {results.metrics.confusion_matrix?.flatMap((row: number[], rowIndex: number) =>
                  row.map((value, columnIndex) => (
                    <div key={`${rowIndex}-${columnIndex}`} className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
                      <div className="text-xs uppercase tracking-[0.22em] text-slate-400">
                        {rowIndex === 0 ? 'Actual negative' : 'Actual positive'} / {columnIndex === 0 ? 'Pred negative' : 'Pred positive'}
                      </div>
                      <div className="mt-3 font-display text-2xl font-semibold text-white">{value}</div>
                    </div>
                  )),
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4 text-sm text-slate-300">
              <div>MAE: {formatNumber(results.metrics.mae, 3)}</div>
              <div>RMSE: {formatNumber(results.metrics.rmse, 3)}</div>
              <div>R2: {formatNumber(results.metrics.r2, 3)}</div>
              <div>Target column: {results.target_column}</div>
            </div>
          )}
        </Card>
      </div>

      <Card title="Comparison table" description="Scores across the models included in the last experiment">
        <DataTable
          columns={['label', 'training_time_seconds', 'device', 'score_for_selection']}
          rows={results.comparison}
          compact
        />
      </Card>
    </div>
  )
}
