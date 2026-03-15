export const formatNumber = (value: number | null | undefined, digits = 0) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-'
  }
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value)
}

export const formatPercent = (value: number | null | undefined, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-'
  }
  return `${(value * 100).toFixed(digits)}%`
}

export const formatMetricLabel = (label: string) =>
  label
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())

export const formatDateTime = (value?: string | null) => {
  if (!value) return '-'
  return new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export const recommendationTone = (value?: string | null) => {
  switch (value) {
    case 'strong_hire':
      return 'emerald'
    case 'shortlist':
      return 'cyan'
    case 'maybe':
      return 'amber'
    case 'reject':
      return 'rose'
    default:
      return 'slate'
  }
}
