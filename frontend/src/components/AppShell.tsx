import clsx from 'clsx'
import {
  AreaChart,
  BookOpenText,
  BrainCircuit,
  CircleGauge,
  Database,
  Layers3,
  LineChart,
  Menu,
  MoonStar,
  Sparkles,
  SunMedium,
  Trophy,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'

import { formatDateTime } from '../lib/format'
import type { DeviceInfo, TrainingStatus } from '../types/api'
import { StatusBadge } from './StatusBadge'

const navigation = [
  { to: '/', label: 'Dashboard', icon: AreaChart },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/training', label: 'Training Lab', icon: BrainCircuit },
  { to: '/performance', label: 'Performance', icon: LineChart },
  { to: '/ranking', label: 'Ranking', icon: Trophy },
  { to: '/playground', label: 'Playground', icon: Sparkles },
  { to: '/compare', label: 'Compare', icon: Layers3 },
]

export const AppShell = ({
  children,
  theme,
  onToggleTheme,
  trainingStatus,
  deviceInfo,
  streamConnected,
}: {
  children: ReactNode
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  trainingStatus: TrainingStatus | null
  deviceInfo: DeviceInfo | null
  streamConnected: boolean
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const showBackLink = location.pathname.startsWith('/candidate/')

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 transition-colors dark:bg-[#07111f] dark:text-slate-100">
      <div className="fixed inset-0 -z-10 bg-mesh opacity-90 dark:bg-mesh" />
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_20%_20%,rgba(255,255,255,0.06),transparent_26%),radial-gradient(circle_at_80%_0%,rgba(41,255,201,0.08),transparent_30%),linear-gradient(180deg,rgba(7,17,31,0.8),rgba(7,17,31,1))]" />

      <div className="mx-auto flex min-h-screen max-w-[1680px] gap-6 px-4 py-4 md:px-6">
        <aside
          className={clsx(
            'fixed inset-y-4 left-4 z-40 w-[290px] rounded-[2rem] border border-white/10 bg-ink-950/90 p-6 shadow-soft backdrop-blur-2xl transition-transform md:sticky md:top-4 md:block md:h-[calc(100vh-2rem)] md:translate-x-0',
            sidebarOpen ? 'translate-x-0' : '-translate-x-[120%]',
          )}
        >
          <div className="flex items-center justify-between">
            <Link to="/" className="space-y-2">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200">
                <CircleGauge className="h-3.5 w-3.5" />
                HR Intelligence
              </div>
              <div>
                <div className="font-display text-2xl font-semibold text-white">Resume AI</div>
                <div className="text-sm text-slate-400">Recruitment analytics platform</div>
              </div>
            </Link>
            <button className="rounded-full border border-white/10 p-2 text-slate-300 md:hidden" onClick={() => setSidebarOpen(false)}>
              <Menu className="h-4 w-4" />
            </button>
          </div>

          <div className="mt-8 rounded-[1.5rem] border border-white/10 bg-white/5 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs uppercase tracking-[0.22em] text-slate-400">Model status</div>
                <div className="mt-2 font-display text-lg font-semibold text-white">
                  {trainingStatus?.active_model_name || 'No active model'}
                </div>
              </div>
              <StatusBadge
                label={trainingStatus?.status || 'idle'}
                tone={trainingStatus?.status === 'trained' ? 'emerald' : trainingStatus?.status === 'training' ? 'cyan' : trainingStatus?.status === 'failed' ? 'rose' : 'slate'}
              />
            </div>
            <div className="mt-4 space-y-2 text-sm text-slate-400">
              <div>Task: {trainingStatus?.task_type || '-'}</div>
              <div>Updated: {formatDateTime(trainingStatus?.finished_at || trainingStatus?.started_at)}</div>
              <div>Device: {deviceInfo?.gpu_name || deviceInfo?.preferred_training_device || '-'}</div>
              <div>VRAM: {deviceInfo?.total_gpu_memory_gb ? `${deviceInfo.total_gpu_memory_gb} GB` : '-'}</div>
              <div className="flex items-center gap-2">
                Stream:
                <StatusBadge label={streamConnected ? 'live' : 'reconnect'} tone={streamConnected ? 'emerald' : 'amber'} />
              </div>
            </div>
          </div>

          <nav className="mt-8 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setSidebarOpen(false)}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium transition',
                      isActive
                        ? 'bg-cyan-400/15 text-white shadow-glow'
                        : 'text-slate-400 hover:bg-white/5 hover:text-white',
                    )
                  }
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </NavLink>
              )
            })}
            {showBackLink && (
              <NavLink to="/ranking" className="flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium text-slate-400 transition hover:bg-white/5 hover:text-white">
                <BookOpenText className="h-4 w-4" />
                Back to Ranking
              </NavLink>
            )}
          </nav>
        </aside>

        <main className="min-w-0 flex-1">
          <div className="mb-6 flex items-center justify-between gap-4 rounded-[1.75rem] border border-white/10 bg-white/5 px-5 py-4 shadow-soft backdrop-blur-xl">
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-2 text-sm text-slate-200 md:hidden"
              onClick={() => setSidebarOpen((value) => !value)}
            >
              <Menu className="h-4 w-4" />
              Menu
            </button>
            <div className="hidden text-sm text-slate-400 md:block">
              Talent screening, ranking, explainability, and experiment tracking in one control center.
            </div>
            <button
              type="button"
              onClick={onToggleTheme}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 transition hover:border-cyan-300/40 hover:text-white"
            >
              {theme === 'dark' ? <SunMedium className="h-4 w-4" /> : <MoonStar className="h-4 w-4" />}
              {theme === 'dark' ? 'Light' : 'Dark'} mode
            </button>
          </div>
          <div className="space-y-6">{children}</div>
        </main>
      </div>
    </div>
  )
}
