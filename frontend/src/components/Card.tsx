import clsx from 'clsx'
import type { PropsWithChildren, ReactNode } from 'react'

interface CardProps extends PropsWithChildren {
  title?: string
  description?: string
  action?: ReactNode
  className?: string
}

export const Card = ({ title, description, action, className, children }: CardProps) => (
  <section
    className={clsx(
      'rounded-[1.75rem] border border-white/10 bg-white/5 p-5 shadow-soft backdrop-blur-xl transition-colors dark:bg-white/5',
      className,
    )}
  >
    {(title || action || description) && (
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          {title && <h3 className="font-display text-lg font-semibold text-slate-50">{title}</h3>}
          {description && <p className="mt-1 text-sm text-slate-400">{description}</p>}
        </div>
        {action}
      </div>
    )}
    {children}
  </section>
)
