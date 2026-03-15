export const LoadingState = ({ label = 'Loading data...' }: { label?: string }) => (
  <div className="flex min-h-[180px] items-center justify-center rounded-[1.75rem] border border-dashed border-white/10 bg-white/5 text-sm text-slate-400">
    <div className="flex items-center gap-3">
      <span className="h-3 w-3 animate-pulse rounded-full bg-cyan-300" />
      <span>{label}</span>
    </div>
  </div>
)

export const EmptyState = ({
  title,
  description,
}: {
  title: string
  description: string
}) => (
  <div className="flex min-h-[180px] flex-col items-center justify-center rounded-[1.75rem] border border-dashed border-white/10 bg-white/5 px-6 text-center">
    <h3 className="font-display text-xl font-semibold text-white">{title}</h3>
    <p className="mt-3 max-w-xl text-sm text-slate-400">{description}</p>
  </div>
)
