import { useEffect, useRef } from 'react'

export const usePolling = (callback: () => void | Promise<void>, delay: number, enabled = true) => {
  const callbackRef = useRef(callback)

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    if (!enabled) return undefined

    void callbackRef.current()
    const timer = window.setInterval(() => {
      void callbackRef.current()
    }, delay)

    return () => window.clearInterval(timer)
  }, [delay, enabled])
}
