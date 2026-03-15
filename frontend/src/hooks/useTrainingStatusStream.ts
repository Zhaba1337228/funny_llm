import { useEffect, useRef, useState } from 'react'

import { client } from '../lib/api'
import type { TrainingStatus } from '../types/api'

const resolveWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/api/ws/training-status`
}

export const useTrainingStatusStream = () => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [streamConnected, setStreamConnected] = useState(false)
  const reconnectRef = useRef<number | null>(null)
  const streamConnectedRef = useRef(false)

  useEffect(() => {
    streamConnectedRef.current = streamConnected
  }, [streamConnected])

  useEffect(() => {
    let cancelled = false
    let socket: WebSocket | null = null

    const fetchSnapshot = async () => {
      const snapshot = await client.trainingStatus().catch(() => null)
      if (!cancelled && snapshot) {
        setTrainingStatus(snapshot)
      }
    }

    const connect = () => {
      if (cancelled) return
      try {
        socket = new WebSocket(resolveWebSocketUrl())
      } catch {
        setStreamConnected(false)
        reconnectRef.current = window.setTimeout(connect, 2500)
        return
      }

      socket.onopen = () => {
        if (cancelled) return
        setStreamConnected(true)
        void fetchSnapshot()
      }

      socket.onmessage = (event) => {
        if (cancelled) return
        try {
          const payload = JSON.parse(event.data) as TrainingStatus
          setTrainingStatus(payload)
        } catch {
          // Ignore malformed frames and keep the connection alive.
        }
      }

      socket.onerror = () => {
        setStreamConnected(false)
      }

      socket.onclose = () => {
        if (cancelled) return
        setStreamConnected(false)
        reconnectRef.current = window.setTimeout(connect, 2500)
      }
    }

    void fetchSnapshot()
    connect()

    const fallbackPoll = window.setInterval(() => {
      void fetchSnapshot()
    }, 2500)

    return () => {
      cancelled = true
      setStreamConnected(false)
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current)
      }
      window.clearInterval(fallbackPoll)
      socket?.close()
    }
  }, [])

  return { trainingStatus, streamConnected }
}
