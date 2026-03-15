import type { ReactNode } from 'react'

export const SectionHeader = ({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string
  title: string
  description?: string
  action?: ReactNode
}) => (
  <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
    <div className="space-y-2">
      {eyebrow && <div className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-300">{eyebrow}</div>}
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-white">{title}</h1>
        {description && <p className="mt-2 max-w-3xl text-sm text-slate-400">{description}</p>}
      </div>
    </div>
    {action}
  </div>
)
