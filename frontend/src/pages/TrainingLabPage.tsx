import clsx from 'clsx'
import { Cpu, Play, Square, WandSparkles } from 'lucide-react'
import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'

import { Card } from '../components/Card'
import { EmptyState, LoadingState } from '../components/LoadingState'
import { SectionHeader } from '../components/SectionHeader'
import { StatusBadge } from '../components/StatusBadge'
import { client, getApiErrorMessage } from '../lib/api'
import { formatDateTime, formatMetricLabel } from '../lib/format'
import type { DatasetOverviewResponse, DeviceInfo, ModelListResponse, TrainingProfile, TrainingStatus } from '../types/api'

export const TrainingLabPage = ({
  trainingStatus,
  deviceInfo,
}: {
  trainingStatus: TrainingStatus | null
  deviceInfo: DeviceInfo | null
}) => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetOverviewResponse | null>(null)
  const [models, setModels] = useState<ModelListResponse | null>(null)
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification')
  const [trainingProfile, setTrainingProfile] = useState<TrainingProfile>('balanced')
  const [modelName, setModelName] = useState('random_forest')
  const [targetColumn, setTargetColumn] = useState('')
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  const [modelsToCompare, setModelsToCompare] = useState<string[]>(['random_forest', 'hist_gradient_boosting', 'torch_mlp'])
  const [testSize, setTestSize] = useState(0.2)
  const [validationSize, setValidationSize] = useState(0.2)
  const [epochs, setEpochs] = useState(24)
  const [batchSize, setBatchSize] = useState(1024)
  const [learningRate, setLearningRate] = useState(0.001)
  const [nEstimators, setNEstimators] = useState(220)
  const [maxDepth, setMaxDepth] = useState(14)
  const [resumeFromRunId, setResumeFromRunId] = useState('')
  const [resumeRounds, setResumeRounds] = useState(16)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      const [dataset, modelCatalog] = await Promise.all([client.datasetInfo(), client.modelsList()])
      if (!mounted) return
      setDatasetInfo(dataset)
      setModels(modelCatalog)
      const defaultTask = dataset.overview.available_tasks.includes('classification') ? 'classification' : 'regression'
      const defaultProfile: TrainingProfile =
        deviceInfo?.cuda_available && (deviceInfo.gpu_count || 0) >= 2
          ? 'server_max'
          : deviceInfo?.cuda_available
            ? 'max_accuracy'
            : 'balanced'
      setTaskType(defaultTask)
      setTrainingProfile(defaultProfile)
      setTargetColumn(dataset.dataset.target_columns[defaultTask] ?? '')
      setSelectedFeatures([...dataset.dataset.numeric_columns, ...dataset.dataset.categorical_columns])
      const preset = resolveTrainingPreset(defaultProfile, defaultTask, modelCatalog.catalog.map((model) => model.name))
      setModelName(preset.modelName)
      setModelsToCompare(preset.modelsToCompare)
      setNEstimators(preset.nEstimators)
      setMaxDepth(preset.maxDepth)
      setLearningRate(preset.learningRate)
      setEpochs(preset.epochs)
      setBatchSize(preset.batchSize)
      const resumableVersion = modelCatalog.saved_versions.find((version) => version.resumable && version.task_type === defaultTask)
      setResumeFromRunId(resumableVersion?.run_id || '')
      setLoading(false)
    }
    void load()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!datasetInfo) return
    setTargetColumn(datasetInfo.dataset.target_columns[taskType] ?? '')
  }, [datasetInfo, taskType])

  useEffect(() => {
    if (!models || !datasetInfo) return
    if (deviceInfo?.cuda_available && (deviceInfo.gpu_count || 0) >= 2 && trainingProfile === 'balanced') {
      setTrainingProfile('server_max')
    }
  }, [datasetInfo, deviceInfo?.cuda_available, deviceInfo?.gpu_count, models, trainingProfile])

  useEffect(() => {
    if (!models) return
    const preset = resolveTrainingPreset(trainingProfile, taskType, models.catalog.map((model) => model.name))
    setModelName(preset.modelName)
    setModelsToCompare(preset.modelsToCompare)
    setNEstimators(preset.nEstimators)
    setMaxDepth(preset.maxDepth)
    setLearningRate(preset.learningRate)
    setEpochs(preset.epochs)
    setBatchSize(preset.batchSize)
  }, [trainingProfile, taskType, models])

  if (loading || !datasetInfo || !models) {
    return <LoadingState label="Preparing training lab..." />
  }

  const modelOptions = models.catalog.filter((model) => model.task_types.includes(taskType))

  const toggleFeature = (feature: string) => {
    setSelectedFeatures((current) =>
      current.includes(feature) ? current.filter((item) => item !== feature) : [...current, feature],
    )
  }

  const toggleCompareModel = (candidate: string) => {
    setModelsToCompare((current) =>
      current.includes(candidate) ? current.filter((item) => item !== candidate) : [...current, candidate],
    )
  }

  const resumableVersions = models.saved_versions.filter((version) => version.resumable && version.task_type === taskType)

  const startTraining = async () => {
    try {
      setFeedback(null)
      await client.startTraining({
        task_type: taskType,
        training_profile: trainingProfile,
        model_name: modelName,
        target_column: targetColumn,
        feature_columns: selectedFeatures,
        models_to_compare: modelsToCompare,
        test_size: testSize,
        validation_size: validationSize,
        hyperparameters: {
          n_estimators: nEstimators,
          max_depth: maxDepth,
          learning_rate: learningRate,
        },
        neural_net: {
          epochs,
          batch_size: batchSize,
          lr: learningRate,
        },
        save_as_best: true,
      })
      setFeedback('Training started successfully.')
    } catch (error) {
      setFeedback(getApiErrorMessage(error))
    }
  }

  const continueTraining = async () => {
    if (!resumeFromRunId) {
      setFeedback('Choose a saved run to continue.')
      return
    }
    try {
      setFeedback(null)
      await client.startTraining({
        resume_from_run_id: resumeFromRunId,
        resume_rounds: resumeRounds,
        task_type: taskType,
        training_profile: trainingProfile,
        save_as_best: true,
      })
      setFeedback('Resume training started successfully.')
    } catch (error) {
      setFeedback(getApiErrorMessage(error))
    }
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Training lab"
        title="Orchestrate model experiments"
        description="Choose the task, define features, compare classical models against the GPU-capable neural net, and watch the live training stream."
        action={
          <div className="flex items-center gap-3">
            <StatusBadge
              label={trainingStatus?.status || 'idle'}
              tone={trainingStatus?.status === 'training' ? 'cyan' : trainingStatus?.status === 'trained' ? 'emerald' : 'slate'}
            />
            {trainingStatus?.status === 'training' ? (
              <button
                type="button"
                onClick={() =>
                  void client
                    .stopTraining()
                    .then(() => setFeedback('Stop requested. The current fit will finish before shutdown.'))
                    .catch((error) => setFeedback(getApiErrorMessage(error)))
                }
                className="inline-flex items-center gap-2 rounded-full border border-rose-400/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-200 transition hover:bg-rose-500/20"
              >
                <Square className="h-4 w-4" />
                Stop
              </button>
            ) : (
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => void continueTraining()}
                  disabled={!resumeFromRunId}
                  className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-100 transition hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Play className="h-4 w-4" />
                  Continue training
                </button>
                <button
                  type="button"
                  onClick={() => void startTraining()}
                  className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-500/20"
                >
                  <Play className="h-4 w-4" />
                  Start training
                </button>
              </div>
            )}
          </div>
        }
      />

      {feedback && (
        <Card className="border-cyan-400/20 bg-cyan-500/10">
          <div className="text-sm text-cyan-50">{feedback}</div>
        </Card>
      )}

      <div className="grid gap-4 xl:grid-cols-[1.2fr,0.8fr]">
        <Card title="Experiment configuration" description="Set task mode, model family, feature scope, and fit parameters">
          <div className="grid gap-4 lg:grid-cols-2">
            <Control label="Task type">
              <select value={taskType} onChange={(event) => setTaskType(event.target.value as 'classification' | 'regression')} className={inputClassName}>
                {datasetInfo.overview.available_tasks.map((task) => (
                  <option key={task} value={task}>
                    {formatMetricLabel(task)}
                  </option>
                ))}
              </select>
            </Control>
            <Control label="Training profile">
              <select value={trainingProfile} onChange={(event) => setTrainingProfile(event.target.value as TrainingProfile)} className={inputClassName}>
                {models.training_profiles.map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.label}
                  </option>
                ))}
              </select>
            </Control>
            <Control label="Primary model">
              <select value={modelName} onChange={(event) => setModelName(event.target.value)} className={inputClassName}>
                {modelOptions.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.label}
                  </option>
                ))}
              </select>
            </Control>
            <Control label="Target column">
              <select value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)} className={inputClassName}>
                <option value={datasetInfo.dataset.target_columns[taskType] ?? ''}>
                  {datasetInfo.dataset.target_columns[taskType] ?? 'Unavailable'}
                </option>
              </select>
            </Control>
            <Control label="Training device">
              <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200">
                <Cpu className="h-4 w-4 text-cyan-300" />
                {deviceInfo?.gpu_name || deviceInfo?.preferred_training_device || 'cpu'}
              </div>
            </Control>
            <Control label="Test split">
              <input value={testSize} min={0.05} max={0.4} step={0.05} type="number" onChange={(event) => setTestSize(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Validation split">
              <input value={validationSize} min={0.05} max={0.4} step={0.05} type="number" onChange={(event) => setValidationSize(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Number of trees / iterations">
              <input value={nEstimators} min={50} step={10} type="number" onChange={(event) => setNEstimators(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Max depth">
              <input value={maxDepth} min={2} step={1} type="number" onChange={(event) => setMaxDepth(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Epochs">
              <input value={epochs} min={5} step={1} type="number" onChange={(event) => setEpochs(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Batch size">
              <input value={batchSize} min={128} step={128} type="number" onChange={(event) => setBatchSize(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Learning rate">
              <input value={learningRate} min={0.0001} max={0.1} step={0.0005} type="number" onChange={(event) => setLearningRate(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Resume from saved run">
              <select value={resumeFromRunId} onChange={(event) => setResumeFromRunId(event.target.value)} className={inputClassName}>
                <option value="">Choose saved run</option>
                {resumableVersions.map((version) => (
                  <option key={version.run_id} value={version.run_id}>
                    {version.model_name} - {version.run_id}
                  </option>
                ))}
              </select>
            </Control>
            <Control label="Additional epochs / iterations">
              <input value={resumeRounds} min={1} step={1} type="number" onChange={(event) => setResumeRounds(Number(event.target.value))} className={inputClassName} />
            </Control>
            <Control label="Auto-save best model">
              <div className="rounded-2xl border border-emerald-400/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
                Enabled
              </div>
            </Control>
          </div>

          <div className="mt-6">
            <div className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Features</div>
            <div className="flex flex-wrap gap-2">
              {[...datasetInfo.dataset.numeric_columns, ...datasetInfo.dataset.categorical_columns].map((feature) => (
                <button
                  key={feature}
                  type="button"
                  onClick={() => toggleFeature(feature)}
                  className={clsx(
                    'rounded-full border px-3 py-2 text-sm transition',
                    selectedFeatures.includes(feature)
                      ? 'border-cyan-300/40 bg-cyan-500/10 text-cyan-100'
                      : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20',
                  )}
                >
                  {formatMetricLabel(feature)}
                </button>
              ))}
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="Model compare set" description="The lab will benchmark each selected model and keep the strongest one">
            <div className="space-y-3">
              {modelOptions.map((model) => (
                <button
                  key={model.name}
                  type="button"
                  onClick={() => toggleCompareModel(model.name)}
                  className={clsx(
                    'w-full rounded-2xl border p-4 text-left transition',
                    modelsToCompare.includes(model.name)
                      ? 'border-cyan-400/30 bg-cyan-400/10 text-white'
                      : 'border-white/10 bg-white/5 text-slate-300 hover:border-white/20',
                  )}
                >
                  <div className="font-medium">{model.label}</div>
                  <div className="mt-1 text-sm text-slate-400">{model.description}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Runtime stream" description="Live status, timing, and training logs">
            <div className="space-y-4">
              <div className="h-3 overflow-hidden rounded-full bg-slate-900/70">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 transition-all"
                  style={{ width: `${Math.max((trainingStatus?.progress || 0) * 100, 4)}%` }}
                />
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <RuntimeInfo label="Current step" value={trainingStatus?.current_step || 'Idle'} />
                <RuntimeInfo label="Elapsed" value={trainingStatus?.elapsed_seconds ? `${trainingStatus.elapsed_seconds}s` : '-'} />
                <RuntimeInfo label="Run ID" value={trainingStatus?.run_id || '-'} />
                <RuntimeInfo label="Last updated" value={formatDateTime(trainingStatus?.finished_at || trainingStatus?.started_at)} />
              </div>
              <div className="max-h-[260px] space-y-2 overflow-auto rounded-[1.25rem] border border-white/10 bg-slate-950/30 p-4 font-mono text-xs text-slate-300">
                {(trainingStatus?.logs || []).length > 0 ? (
                  trainingStatus?.logs.map((log) => <div key={log}>{log}</div>)
                ) : (
                  <EmptyState title="No training logs yet" description="Start a run to stream epoch updates, validation metrics, and completion messages." />
                )}
              </div>
            </div>
          </Card>

          <Card title="Device readiness" description="Automatic hardware detection for neural training">
            <div className="rounded-[1.5rem] border border-cyan-400/20 bg-cyan-500/10 p-4 text-sm text-slate-200">
              <div className="mb-2 flex items-center gap-2 text-cyan-100">
                <WandSparkles className="h-4 w-4" />
                Neural nets use GPU automatically when CUDA is available.
              </div>
              <div>Preferred device: {deviceInfo?.preferred_training_device || '-'}</div>
              <div>GPU: {deviceInfo?.gpu_name || 'not detected'}</div>
              <div>Total VRAM: {deviceInfo?.total_gpu_memory_gb ? `${deviceInfo.total_gpu_memory_gb} GB` : '-'}</div>
              <div>Training profile: {models.training_profiles.find((profile) => profile.name === trainingProfile)?.description}</div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

const Control = ({ label, children }: { label: string; children: ReactNode }) => (
  <label className="space-y-2">
    <span className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">{label}</span>
    {children}
  </label>
)

const RuntimeInfo = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-[1.25rem] border border-white/10 bg-white/5 p-4">
    <div className="text-xs uppercase tracking-[0.22em] text-slate-400">{label}</div>
    <div className="mt-2 text-sm text-white">{value}</div>
  </div>
)

const inputClassName =
  'w-full rounded-2xl border border-white/10 bg-slate-950/20 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40'

const resolveTrainingPreset = (
  profile: TrainingProfile,
  taskType: 'classification' | 'regression',
  availableModels: string[],
) => {
  const presets: Record<TrainingProfile, {
    modelName: string
    modelsToCompare: string[]
    nEstimators: number
    maxDepth: number
    learningRate: number
    epochs: number
    batchSize: number
  }> = {
    rapid: {
      modelName: 'hist_gradient_boosting',
      modelsToCompare: ['hist_gradient_boosting', 'random_forest'],
      nEstimators: 260,
      maxDepth: 8,
      learningRate: 0.08,
      epochs: 16,
      batchSize: 2048,
    },
    balanced: {
      modelName: taskType === 'classification' ? 'random_forest' : 'hist_gradient_boosting',
      modelsToCompare: ['catboost', 'xgboost', 'hist_gradient_boosting', 'extra_trees', 'torch_mlp'],
      nEstimators: 600,
      maxDepth: 10,
      learningRate: 0.05,
      epochs: 32,
      batchSize: 4096,
    },
    max_accuracy: {
      modelName: 'xgboost',
      modelsToCompare: ['catboost', 'xgboost', 'hist_gradient_boosting', 'extra_trees', 'torch_mlp'],
      nEstimators: 1200,
      maxDepth: 10,
      learningRate: 0.035,
      epochs: 48,
      batchSize: 8192,
    },
    server_max: {
      modelName: 'catboost',
      modelsToCompare: ['catboost', 'xgboost', 'torch_mlp'],
      nEstimators: 1800,
      maxDepth: 12,
      learningRate: 0.025,
      epochs: 72,
      batchSize: 32768,
    },
  }

  const preset = presets[profile]
  const filteredCompare = preset.modelsToCompare.filter((model) => availableModels.includes(model))
  const resolvedModelName = availableModels.includes(preset.modelName) ? preset.modelName : filteredCompare[0] || availableModels[0] || 'random_forest'
  return {
    ...preset,
    modelName: resolvedModelName,
    modelsToCompare: filteredCompare.length > 0 ? filteredCompare : [resolvedModelName],
  }
}
