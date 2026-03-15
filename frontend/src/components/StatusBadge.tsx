import clsx from 'clsx'

const toneClassMap: Record<string, string> = {
  success: 'bg-emerald-500/15 text-emerald-300 ring-emerald-400/30',
  emerald: 'bg-emerald-500/15 text-emerald-300 ring-emerald-400/30',
  cyan: 'bg-cyan-500/15 text-cyan-300 ring-cyan-400/30',
  warning: 'bg-amber-500/15 text-amber-300 ring-amber-400/30',
  amber: 'bg-amber-500/15 text-amber-300 ring-amber-400/30',
  rose: 'bg-rose-500/15 text-rose-300 ring-rose-400/30',
  error: 'bg-rose-500/15 text-rose-300 ring-rose-400/30',
  slate: 'bg-slate-500/15 text-slate-300 ring-slate-400/30',
}

export const StatusBadge = ({ label, tone = 'slate' }: { label: string; tone?: string }) => (
  <span className={clsx('inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ring-1', toneClassMap[tone] || toneClassMap.slate)}>
    {label}
  </span>
)
