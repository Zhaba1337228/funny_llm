import type { Data, Layout } from 'plotly.js'
import Plot from 'react-plotly.js'

import { Card } from './Card'

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
  </Card>
)
