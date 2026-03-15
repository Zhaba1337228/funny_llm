import { Suspense, lazy, useEffect, useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/AppShell'
import { LoadingState } from './components/LoadingState'
import { useTrainingStatusStream } from './hooks/useTrainingStatusStream'
import { client } from './lib/api'
import type { DeviceInfo } from './types/api'

const DashboardPage = lazy(async () => import('./pages/DashboardPage').then((module) => ({ default: module.DashboardPage })))
const DatasetPage = lazy(async () => import('./pages/DatasetPage').then((module) => ({ default: module.DatasetPage })))
const TrainingLabPage = lazy(async () => import('./pages/TrainingLabPage').then((module) => ({ default: module.TrainingLabPage })))
const PerformancePage = lazy(async () => import('./pages/PerformancePage').then((module) => ({ default: module.PerformancePage })))
const RankingPage = lazy(async () => import('./pages/RankingPage').then((module) => ({ default: module.RankingPage })))
const CandidateDetailPage = lazy(async () => import('./pages/CandidateDetailPage').then((module) => ({ default: module.CandidateDetailPage })))
const PlaygroundPage = lazy(async () => import('./pages/PlaygroundPage').then((module) => ({ default: module.PlaygroundPage })))
const ComparePage = lazy(async () => import('./pages/ComparePage').then((module) => ({ default: module.ComparePage })))

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const stored = window.localStorage.getItem('resume-ai-theme')
    return stored === 'light' ? 'light' : 'dark'
  })
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null)
  const { trainingStatus, streamConnected } = useTrainingStatusStream()

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    window.localStorage.setItem('resume-ai-theme', theme)
  }, [theme])

  useEffect(() => {
    void client.deviceInfo().then(setDeviceInfo).catch(() => null)
  }, [])

  return (
    <BrowserRouter>
      <AppShell
        theme={theme}
        onToggleTheme={() => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))}
        trainingStatus={trainingStatus}
        deviceInfo={deviceInfo}
        streamConnected={streamConnected}
      >
        <Suspense fallback={<LoadingState label="Loading workspace..." />}>
          <Routes>
            <Route path="/" element={<DashboardPage trainingStatus={trainingStatus} deviceInfo={deviceInfo} />} />
            <Route path="/dataset" element={<DatasetPage />} />
            <Route path="/training" element={<TrainingLabPage trainingStatus={trainingStatus} deviceInfo={deviceInfo} />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/ranking" element={<RankingPage />} />
            <Route path="/candidate/:candidateId" element={<CandidateDetailPage />} />
            <Route path="/playground" element={<PlaygroundPage />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </Suspense>
      </AppShell>
    </BrowserRouter>
  )
}

export default App
