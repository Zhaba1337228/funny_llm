import type { Data, Layout } from 'plotly.js'
import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

import { Card } from './Card'

const resolvedPlotly = (Plotly as { default?: unknown })?.default ?? Plotly
const resolvedFactory =
  (createPlotlyComponent as unknown as { default?: ((plotly: unknown) => any) })?.default ?? createPlotlyComponent

const Plot = typeof resolvedFactory === 'function' ? resolvedFactory(resolvedPlotly) : null

export const PlotPanel = ({
  title,
  description,
  data,
  layout,
}: {
  title: string
  description?: string
  data: Data[]
  layout?: Partial<Layout>
}) => (
  <Card title={title} description={description} className="overflow-hidden">
    {Plot ? (
      <Plot
        data={data}
        layout={{
          autosize: true,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          margin: { l: 36, r: 12, t: 12, b: 36 },
          font: { color: '#cbd5e1' },
          xaxis: { gridcolor: 'rgba(148, 163, 184, 0.12)' },
          yaxis: { gridcolor: 'rgba(148, 163, 184, 0.12)' },
          ...layout,
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height: '320px' }}
      />
    ) : (
      <div className="flex h-[320px] items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-white/5 text-sm text-slate-400">
        Plot renderer is unavailable in this build.
      </div>
    )}
  </Card>
)
