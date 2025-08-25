import { useState, useEffect, useCallback } from 'react'
import { JudgeBuildersService, JudgesService, LabelingService, ExperimentsService, AlignmentService } from '@/fastapi_client'
import type { JudgeResponse, ExamplesResponse, LabelingProgress } from '@/fastapi_client'

// Hook for fetching judge details
export function useJudge(judgeId: string | undefined) {
  const [judge, setJudge] = useState<JudgeResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!judgeId) {
      setLoading(false)
      return
    }

    const fetchJudge = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await JudgeBuildersService.getJudgeBuilderApiJudgeBuildersJudgeIdGet(judgeId)
        setJudge(response)
      } catch (err) {
        console.error('Error fetching judge:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch judge')
      } finally {
        setLoading(false)
      }
    }

    fetchJudge()
  }, [judgeId])

  return { judge, loading, error, refetch: () => {
    if (judgeId) {
      JudgeBuildersService.getJudgeBuilderApiJudgeBuildersJudgeIdGet(judgeId).then(setJudge).catch(console.error)
    }
  }}
}

// Hook for fetching judge examples
export function useJudgeExamples(judgeId: string | undefined, includeJudgeResults: boolean = false) {
  const [examples, setExamples] = useState<ExamplesResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!judgeId) {
      setLoading(false)
      return
    }

    const fetchExamples = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await LabelingService.getExamplesApiLabelingJudgeIdExamplesGet(judgeId, includeJudgeResults)
        setExamples(response)
      } catch (err) {
        console.error('Error fetching examples:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch examples')
      } finally {
        setLoading(false)
      }
    }

    fetchExamples()
  }, [judgeId, includeJudgeResults])

  return { examples, loading, error, refetch: () => {
    if (judgeId) {
      setLoading(true)
      setError(null)
      LabelingService.getExamplesApiLabelingJudgeIdExamplesGet(judgeId, includeJudgeResults)
        .then(setExamples)
        .catch(err => {
          console.error(err)
          setError(err instanceof Error ? err.message : 'Failed to fetch examples')
        })
        .finally(() => setLoading(false))
    }
  }}
}

// Hook for fetching labeling progress
export function useLabelingProgress(judgeId: string | undefined) {
  const [progress, setProgress] = useState<LabelingProgress | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!judgeId) {
      setLoading(false)
      return
    }

    const fetchProgress = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await LabelingService.getLabelingProgressApiLabelingJudgeIdLabelingProgressGet(judgeId)
        setProgress(response)
      } catch (err) {
        console.error('Error fetching labeling progress:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch labeling progress')
      } finally {
        setLoading(false)
      }
    }

    fetchProgress()
  }, [judgeId])

  return { progress, loading, error, refetch: () => {
    if (judgeId) {
      setLoading(true)
      setError(null)
      LabelingService.getLabelingProgressApiLabelingJudgeIdLabelingProgressGet(judgeId)
        .then(setProgress)
        .catch(err => {
          console.error(err)
          setError(err instanceof Error ? err.message : 'Failed to fetch labeling progress')
        })
        .finally(() => setLoading(false))
    }
  }}
}

// Hook for running alignment
export function useAlignment() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [needsRefresh, setNeedsRefresh] = useState(false)

  const runAlignment = async (judgeId: string, experimentId: string) => {
    try {
      setLoading(true)
      setError(null)
      setNeedsRefresh(false)
      const response = await AlignmentService.runAlignmentApiAlignmentJudgeIdAlignPost(judgeId)
      return response
    } catch (err) {
      console.error('Error running alignment:', err)
      
      // Check if this is a 504 timeout error
      const isTimeout = err instanceof Error && (
        err.message.includes('504') || 
        err.message.includes('Gateway Timeout') ||
        err.message.includes('timeout')
      )
      
      if (isTimeout) {
        
        // Poll for alignment completion using exponential backoff
        const pollForCompletion = async (attempt: number = 0, maxRetries: number = 7): Promise<any> => {
          if (attempt >= maxRetries) {
            throw new Error('Alignment timeout: maximum polling attempts reached')
          }
          
          try {
            // Try to get alignment comparison data - if this succeeds, alignment completed
            const comparisonResult = await AlignmentService.getAlignmentComparisonApiAlignmentJudgeIdAlignmentComparisonGet(judgeId)
            return comparisonResult
          } catch (pollErr) {
            
            if (attempt < maxRetries - 1) {
              // Exponential backoff: 2^attempt * 2000ms (2s, 4s, 8s, 16s, 32s)
              const delay = Math.pow(2, attempt) * 2000
              await new Promise(resolve => setTimeout(resolve, delay))
              return pollForCompletion(attempt + 1, maxRetries)
            } else {
              throw new Error('Alignment timeout: polling unsuccessful after maximum attempts')
            }
          }
        }
        
        try {
          await pollForCompletion()
          // If polling succeeds, return a success response
          return { success: true, message: 'Alignment completed successfully (detected via polling)' }
        } catch (pollError) {
          console.error('[Alignment Polling] Failed to detect alignment completion:', pollError)
          setError('Alignment may have completed but polling failed')
          setNeedsRefresh(true)
          throw pollError
        }
      } else {
        setError(err instanceof Error ? err.message : 'Failed to run alignment')
        throw err
      }
    } finally {
      setLoading(false)
    }
  }

  const checkAlignmentStatus = async (judgeId: string) => {
    try {
      setLoading(true)
      setError(null)
      setNeedsRefresh(false)
      const response = await AlignmentService.getAlignmentComparisonApiAlignmentJudgeIdAlignmentComparisonGet(judgeId)
      return response
    } catch (err) {
      console.error('Error checking alignment status:', err)
      setError(err instanceof Error ? err.message : 'Failed to check alignment status')
      throw err
    } finally {
      setLoading(false)
    }
  }

  return { runAlignment, checkAlignmentStatus, loading, error, needsRefresh }
}

// Hook for fetching alignment comparison data
export function useAlignmentComparison(judgeId: string | undefined) {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const [hasFetched, setHasFetched] = useState(false)

  const fetchComparison = useCallback(async (maxRetries = 5) => {
    if (!judgeId) {
      return
    }

    setHasFetched(true)
    setLoading(true)
    setError(null)
    
    const attemptFetch = async (attempt: number): Promise<any> => {
      try {
        const response = await AlignmentService.getAlignmentComparisonApiAlignmentJudgeIdAlignmentComparisonGet(judgeId)
        setData(response)
        setRetryCount(0) // Reset retry count on success
        setLoading(false)
        return response
      } catch (err) {
        console.error(`[Alignment Debug] Error fetching alignment comparison for judge ${judgeId} (attempt ${attempt + 1}):`, err)
        console.error(`[Alignment Debug] Error details:`, {
          name: err instanceof Error ? err.name : 'Unknown',
          message: err instanceof Error ? err.message : String(err),
          stack: err instanceof Error ? err.stack : undefined,
          judgeId,
          attempt: attempt + 1,
          maxRetries
        })
        
        if (attempt < maxRetries - 1) {
          // Wait before retrying (exponential backoff: 1s, 2s, 4s, 8s, 16s)
          const delay = Math.pow(2, attempt) * 1000
          await new Promise(resolve => setTimeout(resolve, delay))
          setRetryCount(attempt + 1)
          return attemptFetch(attempt + 1)
        } else {
          // Max retries reached
          const errorMessage = err instanceof Error ? err.message : 'Failed to fetch alignment comparison'
          console.error(`[Alignment Debug] Max retries (${maxRetries}) reached for judge ${judgeId}. Final error:`, errorMessage)
          setError(errorMessage)
          setRetryCount(maxRetries)
          setLoading(false)
          throw new Error(errorMessage)
        }
      }
    }

    return attemptFetch(0)
  }, [])

  const resetData = useCallback(() => {
    setData(null)
    setError(null)
    setHasFetched(false)
    setRetryCount(0)
  }, [judgeId])

  return { data, loading, error, fetchComparison, resetData, retryCount, hasFetched }
}

// Hook for fetching experiment traces
export function useExperimentTraces(experimentId: string | undefined, runId?: string) {
  const [traces, setTraces] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!experimentId) {
      setLoading(false)
      return
    }

    const fetchTraces = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await ExperimentsService.getExperimentTracesApiExperimentsExperimentIdTracesGet(experimentId, runId)
        setTraces(response.traces || [])
      } catch (err) {
        console.error('Error fetching traces:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch traces')
      } finally {
        setLoading(false)
      }
    }

    fetchTraces()
  }, [experimentId, runId])

  return { traces, loading, error, refetch: () => {
    if (experimentId) {
      ExperimentsService.getExperimentTracesApiExperimentsExperimentIdTracesGet(experimentId, runId)
        .then(response => setTraces(response.traces || []))
        .catch(console.error)
    }
  }}
}