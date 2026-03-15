import type { ErrorInfo, ReactNode } from 'react'
import { Component } from 'react'

type Props = {
  children: ReactNode
}

type State = {
  hasError: boolean
  message: string
}

export class AppErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    message: '',
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      message: error.message || 'Unknown frontend error',
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('AppErrorBoundary caught an error', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-[1.75rem] border border-rose-400/20 bg-rose-500/10 p-6 text-slate-100">
          <div className="text-sm font-semibold uppercase tracking-[0.22em] text-rose-200">Frontend error</div>
          <div className="mt-3 text-lg font-semibold">The dashboard hit a runtime error.</div>
          <div className="mt-2 text-sm text-slate-300">{this.state.message}</div>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="mt-4 rounded-full border border-rose-300/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-100 transition hover:bg-rose-500/20"
          >
            Reload page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
