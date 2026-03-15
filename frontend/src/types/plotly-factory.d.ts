declare module 'plotly.js-dist-min' {
  const Plotly: unknown
  export default Plotly
}

declare module 'react-plotly.js/factory' {
  const createPlotlyComponent: (plotly: unknown) => any
  export default createPlotlyComponent
}
