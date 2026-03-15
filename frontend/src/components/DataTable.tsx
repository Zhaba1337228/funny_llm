import clsx from 'clsx'
import { ArrowDownWideNarrow, ArrowUpWideNarrow } from 'lucide-react'

import { formatMetricLabel } from '../lib/format'

interface DataTableProps {
  columns: string[]
  rows: Record<string, any>[]
  sortBy?: string
  sortDir?: 'asc' | 'desc'
  onSort?: (column: string) => void
  onRowClick?: (row: Record<string, any>) => void
  compact?: boolean
}

export const DataTable = ({ columns, rows, sortBy, sortDir, onSort, onRowClick, compact = false }: DataTableProps) => (
  <div className="overflow-hidden rounded-[1.5rem] border border-white/10">
    <div className="max-h-[560px] overflow-auto">
      <table className="min-w-full divide-y divide-white/10">
        <thead className="sticky top-0 z-10 bg-ink-900/95 backdrop-blur">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                <button
                  type="button"
                  className="flex items-center gap-2 transition hover:text-white"
                  onClick={() => onSort?.(column)}
                >
                  <span>{formatMetricLabel(column)}</span>
                  {sortBy === column ? (
                    sortDir === 'asc' ? <ArrowUpWideNarrow className="h-3.5 w-3.5" /> : <ArrowDownWideNarrow className="h-3.5 w-3.5" />
                  ) : null}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5 bg-slate-950/20">
          {rows.map((row, index) => (
            <tr
              key={`${row.candidate_id ?? 'row'}-${index}`}
              className={clsx('transition hover:bg-white/5', onRowClick && 'cursor-pointer')}
              onClick={() => onRowClick?.(row)}
            >
              {columns.map((column) => (
                <td key={column} className={clsx('px-4 text-slate-200', compact ? 'py-2 text-sm' : 'py-3 text-sm')}>
                  {row[column] === null || row[column] === undefined ? '-' : String(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
)
