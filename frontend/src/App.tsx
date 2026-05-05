import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type SitemapGroup = {
  group_id: string
  group_name: string
  url_count: number
  sample_urls: string[]
  source_sitemaps_count: number
}

type SitemapLoadResponse = {
  sitemap_url: string
  total_urls: number
  discovered_sitemaps_count: number
  sample_sitemaps: string[]
  groups: SitemapGroup[]
}

type IngestItem = {
  source_url: string
  title: string
  markdown_length: number
}

type IngestGroupsResponse = {
  ingested_count: number
  skipped_count?: number
  failed_count?: number
  items: IngestItem[]
}

type ChatResponse = {
  answer: string
  retrieved_count: number
  sources: string[]
}

type ChatStreamMetaPayload = {
  retrieval_top_k_used: number
  retrieved_count: number
  sources: string[]
  chunk_ids: string[]
  retrieval_mode: string
  retrieval_confidence_inputs: Record<string, unknown>
  performance: Record<string, unknown>
}

type ChatStreamSseMessage =
  | { type: 'meta'; payload: ChatStreamMetaPayload }
  | { type: 'token'; delta: string }
  | { type: 'done'; answer: string; performance: Record<string, unknown> }
  | { type: 'error'; detail: string }

type DataStatusResponse = {
  raw_count: number
  clean_count: number
  processing_count: number
  enrichment_count?: number
  chunks_count?: number
  raw_dir: string
  clean_dir: string
  processing_dir: string
  enriched_dir?: string
  semantic_chunks_dir?: string
  qdrant_collection?: string
  qdrant_points_count?: number | null
  qdrant_error?: string
}

type ChunkingJobPayload = {
  job_id: string
  status: string
  processed_files: number
  total_files: number
  percent: number
  current_file: string
  error: string | null
  result: {
    files_written: number
    files_skipped: number
    files_failed: number
    chunks_total: number
  } | null
}

type IndexingJobPayload = {
  job_id: string
  status: string
  /** queued → downloading_model → loading_model → qdrant → embedding (then completed / failed / unknown) */
  phase?: string
  processed_files: number
  total_files: number
  /** Qdrant points upserted so far (updates during embedding phase). */
  points_indexed?: number
  percent: number
  current_file: string
  error: string | null
  result: {
    manifests_processed: number
    manifests_failed: number
    points_upserted: number
  } | null
}

type EvaluationResponse = {
  cases_count: number
  retrieval_hit_rate: number
  grounded_hit_rate: number
  hallucination_rate_estimate: number
  cases: Array<{
    query: string
    retrieved_count: number
    retrieval_hit: number
    grounded_hit: number
    matched_term: string | null
    expected_terms: string[]
    answer_preview: string
    sources: string[]
    retrieval_mode?: string
    chunk_ids?: string[]
  }>
}

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [sitemapUrl, setSitemapUrl] = useState('')
  const [groups, setGroups] = useState<SitemapGroup[]>([])
  const [selectedGroupIds, setSelectedGroupIds] = useState<string[]>([])
  const [groupSearch, setGroupSearch] = useState('')
  const [discoveryInfo, setDiscoveryInfo] = useState<{
    totalUrls: number
    discoveredSitemapsCount: number
    sampleSitemaps: string[]
  } | null>(null)
  const [isLoadingGroups, setIsLoadingGroups] = useState(false)
  const [isIngestingGroups, setIsIngestingGroups] = useState(false)
  const [skipScrapeIfExists, setSkipScrapeIfExists] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [ingestError, setIngestError] = useState('')
  const [ingestResult, setIngestResult] = useState<IngestGroupsResponse | null>(null)
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatAnswer, setChatAnswer] = useState('')
  const [chatSources, setChatSources] = useState<string[]>([])
  const [chatError, setChatError] = useState('')
  const [isChatting, setIsChatting] = useState(false)
  const [streamAnswer, setStreamAnswer] = useState('')
  const [streamMeta, setStreamMeta] = useState<ChatStreamMetaPayload | null>(null)
  const [streamStatus, setStreamStatus] = useState('')
  const [streamError, setStreamError] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const streamAbortRef = useRef<AbortController | null>(null)
  const [dataStatus, setDataStatus] = useState<DataStatusResponse | null>(null)
  const [cleanupError, setCleanupError] = useState('')
  const [cleanupMessage, setCleanupMessage] = useState('')
  const [isCleaningData, setIsCleaningData] = useState(false)
  const [isRunningPipelineStep, setIsRunningPipelineStep] = useState(false)
  const [evalInput, setEvalInput] = useState(
    JSON.stringify(
      [
        { query: 'What awards or accreditations does LSBF mention?', expected_terms: ['award', 'accreditation'] },
        { query: 'What is corporate social responsibility at LSBF?', expected_terms: ['corporate social responsibility', 'social'] },
      ],
      null,
      2,
    ),
  )
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResponse | null>(null)
  const [evaluationError, setEvaluationError] = useState('')
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [evalTopK, setEvalTopK] = useState(4)
  const [chunkingJobId, setChunkingJobId] = useState<string | null>(null)
  const [chunkingJob, setChunkingJob] = useState<ChunkingJobPayload | null>(null)
  const [chunkingForce, setChunkingForce] = useState(false)
  const [chunkingError, setChunkingError] = useState('')
  const [indexingJobId, setIndexingJobId] = useState<string | null>(null)
  const [indexingJob, setIndexingJob] = useState<IndexingJobPayload | null>(null)
  const [indexingRecreateCollection, setIndexingRecreateCollection] = useState(false)
  const [indexingError, setIndexingError] = useState('')
  const [isDeletingEmbeddings, setIsDeletingEmbeddings] = useState(false)
  const [embeddingDeleteMessage, setEmbeddingDeleteMessage] = useState('')
  const [embeddingDeleteError, setEmbeddingDeleteError] = useState('')
  const [enrichmentForce, setEnrichmentForce] = useState(false)
  const [isEnriching, setIsEnriching] = useState(false)
  const [enrichmentMessage, setEnrichmentMessage] = useState('')
  const [enrichmentError, setEnrichmentError] = useState('')

  const selectedCount = useMemo(() => selectedGroupIds.length, [selectedGroupIds])

  const refreshDataStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/data/status`)
      if (response.ok) {
        setDataStatus((await response.json()) as DataStatusResponse)
      }
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    if (!chunkingJobId) {
      return undefined
    }
    const url = `${API_BASE_URL}/admin/data/chunking-events/${chunkingJobId}`
    const es = new EventSource(url)
    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as ChunkingJobPayload
        setChunkingJob(payload)
        if (payload.status === 'completed' || payload.status === 'failed' || payload.status === 'unknown') {
          es.close()
          setChunkingJobId(null)
          void refreshDataStatus()
        }
      } catch {
        /* ignore malformed event */
      }
    }
    es.onerror = () => {
      es.close()
      setChunkingJobId((current) => {
        if (current) {
          setChunkingError('Progress stream disconnected. If the API restarted, run semantic chunking again.')
        }
        return null
      })
    }
    return () => {
      es.close()
    }
  }, [chunkingJobId, refreshDataStatus])

  useEffect(() => {
    if (!indexingJobId) {
      return undefined
    }
    const url = `${API_BASE_URL}/admin/data/indexing-events/${indexingJobId}`
    const es = new EventSource(url)
    es.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as IndexingJobPayload
        setIndexingJob(payload)
        if (payload.status === 'completed' || payload.status === 'failed' || payload.status === 'unknown') {
          es.close()
          setIndexingJobId(null)
          void refreshDataStatus()
        }
      } catch {
        /* ignore malformed event */
      }
    }
    es.onerror = () => {
      es.close()
      setIndexingJobId((current) => {
        if (current) {
          setIndexingError('Progress stream disconnected. If the API restarted, run persist again.')
        }
        return null
      })
    }
    return () => {
      es.close()
    }
  }, [indexingJobId, refreshDataStatus])

  const handleLoadSitemap = async () => {
    setLoadError('')
    setIngestError('')
    setIngestResult(null)
    setGroups([])
    setSelectedGroupIds([])
    setDiscoveryInfo(null)
    setIsLoadingGroups(true)

    try {
      const response = await fetch(`${API_BASE_URL}/sitemap/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sitemap_url: sitemapUrl }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as SitemapLoadResponse
      setGroups(payload.groups)
      setDiscoveryInfo({
        totalUrls: payload.total_urls,
        discoveredSitemapsCount: payload.discovered_sitemaps_count,
        sampleSitemaps: payload.sample_sitemaps,
      })
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Failed to load sitemap.')
    } finally {
      setIsLoadingGroups(false)
    }
  }

  const handleToggleGroup = (groupId: string) => {
    setSelectedGroupIds((currentGroupIds) => {
      if (currentGroupIds.includes(groupId)) {
        return currentGroupIds.filter((item) => item !== groupId)
      }
      return [...currentGroupIds, groupId]
    })
  }

  const handleIngestGroups = async () => {
    setIngestError('')
    setIngestResult(null)
    setIsIngestingGroups(true)
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/groups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sitemap_url: sitemapUrl,
          selected_group_ids: selectedGroupIds,
          skip_scrape_if_exists: skipScrapeIfExists,
        }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as IngestGroupsResponse
      setIngestResult(payload)
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : 'Failed to ingest groups.')
    } finally {
      setIsIngestingGroups(false)
    }
  }

  const visibleGroups = useMemo(() => {
    if (!groupSearch.trim()) {
      return groups
    }
    const normalizedSearch = groupSearch.toLowerCase()
    return groups.filter((group) =>
      `${group.group_name} ${group.group_id}`.toLowerCase().includes(normalizedSearch),
    )
  }, [groups, groupSearch])

  const handleSelectAllVisible = () => {
    const visibleGroupIds = visibleGroups.map((group) => group.group_id)
    setSelectedGroupIds((currentGroupIds) =>
      Array.from(new Set([...currentGroupIds, ...visibleGroupIds])),
    )
  }

  const handleClearSelection = () => {
    setSelectedGroupIds([])
  }

  const handleAskQuestion = async () => {
    setChatError('')
    setChatAnswer('')
    setChatSources([])
    setIsChatting(true)
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: chatQuestion, top_k: 4 }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as ChatResponse
      setChatAnswer(payload.answer)
      setChatSources(payload.sources)
    } catch (error) {
      setChatError(error instanceof Error ? error.message : 'Failed to generate answer.')
    } finally {
      setIsChatting(false)
    }
  }

  const handleCancelStream = () => {
    streamAbortRef.current?.abort()
    streamAbortRef.current = null
  }

  const handleAskStream = async () => {
    streamAbortRef.current?.abort()
    const controller = new AbortController()
    streamAbortRef.current = controller

    setStreamError('')
    setStreamAnswer('')
    setStreamMeta(null)
    setStreamStatus('Connecting…')
    setIsStreaming(true)

    const appendToken = (delta: string) => {
      setStreamAnswer((prev) => prev + delta)
    }

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: chatQuestion, top_k: 4 }),
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      if (!response.body) {
        throw new Error('No response body (streaming not supported in this browser).')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const processBlock = (block: string) => {
        const lines = block.split('\n')
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue
          let msg: ChatStreamSseMessage
          try {
            msg = JSON.parse(raw) as ChatStreamSseMessage
          } catch {
            continue
          }
          if (msg.type === 'meta') {
            setStreamMeta(msg.payload)
            setStreamStatus(
              `Retrieval done · mode ${msg.payload.retrieval_mode} · ${msg.payload.retrieved_count} chunk(s). Streaming tokens…`,
            )
          } else if (msg.type === 'token') {
            appendToken(msg.delta)
          } else if (msg.type === 'done') {
            setStreamStatus(
              `Done · llm_generation_s=${(msg.performance.llm_generation_s as number) ?? '?'} · chat_total_s=${(msg.performance.chat_total_s as number) ?? '?'}`,
            )
          } else if (msg.type === 'error') {
            throw new Error(msg.detail)
          }
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() ?? ''
        for (const chunk of chunks) {
          if (chunk.trim()) processBlock(chunk)
        }
      }
      if (buffer.trim()) {
        processBlock(buffer)
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setStreamStatus('Cancelled.')
        return
      }
      setStreamError(error instanceof Error ? error.message : 'Stream failed.')
    } finally {
      setIsStreaming(false)
      streamAbortRef.current = null
    }
  }

  const handleLoadDataStatus = async () => {
    setCleanupError('')
    setCleanupMessage('')
    try {
      const response = await fetch(`${API_BASE_URL}/data/status`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as DataStatusResponse
      setDataStatus(payload)
    } catch (error) {
      setCleanupError(error instanceof Error ? error.message : 'Failed to load data status.')
    }
  }

  const handleCleanupData = async (
    clearRaw: boolean,
    clearClean: boolean,
    clearProcessing: boolean,
    clearEnriched: boolean,
    clearSemanticChunks: boolean,
    confirmLabel: string,
  ) => {
    const isConfirmed = window.confirm(`Confirm cleanup: ${confirmLabel}? This cannot be undone.`)
    if (!isConfirmed) return

    setCleanupError('')
    setCleanupMessage('')
    setIsCleaningData(true)
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/cleanup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          clear_raw: clearRaw,
          clear_clean: clearClean,
          clear_processing: clearProcessing,
          clear_enriched: clearEnriched,
          clear_semantic_chunks: clearSemanticChunks,
        }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as {
        deleted_raw: number
        deleted_clean: number
        deleted_processing: number
        deleted_enriched: number
        deleted_semantic_chunks: number
        counts_after_cleanup: DataStatusResponse
      }
      setDataStatus(payload.counts_after_cleanup)
      setCleanupMessage(
        `Deleted raw=${payload.deleted_raw}, clean=${payload.deleted_clean}, processing=${payload.deleted_processing}, enriched=${payload.deleted_enriched}, semantic_chunks=${payload.deleted_semantic_chunks}`,
      )
    } catch (error) {
      setCleanupError(error instanceof Error ? error.message : 'Cleanup failed.')
    } finally {
      setIsCleaningData(false)
    }
  }

  const handleRunPipelineStep = async (endpoint: 'run-clean' | 'run-processing') => {
    setCleanupError('')
    setCleanupMessage('')
    setIsRunningPipelineStep(true)
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/${endpoint}`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as Record<string, number>
      setCleanupMessage(
        Object.entries(payload)
          .map(([key, value]) => `${key}=${value}`)
          .join(', '),
      )
      await handleLoadDataStatus()
    } catch (error) {
      setCleanupError(error instanceof Error ? error.message : 'Failed to run pipeline step.')
    } finally {
      setIsRunningPipelineStep(false)
    }
  }

  const handleRunEnrichment = async () => {
    setEnrichmentError('')
    setEnrichmentMessage('')
    setIsEnriching(true)
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/run-enrichment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: enrichmentForce, limit: null }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as {
        enriched_count: number
        skipped_count: number
        failed_count: number
      }
      setEnrichmentMessage(
        `Enriched ${payload.enriched_count}, skipped ${payload.skipped_count}, failed ${payload.failed_count}.`,
      )
      await handleLoadDataStatus()
    } catch (error) {
      setEnrichmentError(error instanceof Error ? error.message : 'Enrichment failed.')
    } finally {
      setIsEnriching(false)
    }
  }

  const handleStartChunking = async () => {
    setChunkingError('')
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/run-chunking`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: chunkingForce, limit: null }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = (await response.json()) as { job_id: string; total_files: number }
      setChunkingJobId(data.job_id)
      setChunkingJob({
        job_id: data.job_id,
        status: 'running',
        processed_files: 0,
        total_files: data.total_files,
        percent: 0,
        current_file: '',
        error: null,
        result: null,
      })
    } catch (error) {
      setChunkingError(error instanceof Error ? error.message : 'Chunking job failed to start.')
    }
  }

  const handleStartIndexing = async () => {
    setIndexingError('')
    setEmbeddingDeleteMessage('')
    setEmbeddingDeleteError('')
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/run-indexing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ recreate_collection: indexingRecreateCollection, limit: null }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = (await response.json()) as { job_id: string; total_files: number }
      setIndexingJobId(data.job_id)
      setIndexingJob({
        job_id: data.job_id,
        status: 'running',
        phase: 'queued',
        processed_files: 0,
        total_files: data.total_files,
        points_indexed: 0,
        percent: 0,
        current_file: '',
        error: null,
        result: null,
      })
    } catch (error) {
      setIndexingError(error instanceof Error ? error.message : 'Indexing job failed to start.')
    }
  }

  const handleDeleteEmbeddings = async () => {
    setEmbeddingDeleteError('')
    setEmbeddingDeleteMessage('')
    if (
      !window.confirm(
        'Delete all vector embeddings in the Qdrant collection? This removes every point; chunk JSON files on disk are not affected.',
      )
    ) {
      return
    }
    setIsDeletingEmbeddings(true)
    try {
      const response = await fetch(`${API_BASE_URL}/admin/data/delete-embeddings`, { method: 'POST' })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as { removed_points: number | null; qdrant_points_count: number }
      setEmbeddingDeleteMessage(
        payload.removed_points != null
          ? `Removed ${payload.removed_points} point(s) from Qdrant.`
          : 'Qdrant collection recreated (previous point count unavailable).',
      )
      try {
        const statusResponse = await fetch(`${API_BASE_URL}/data/status`)
        if (statusResponse.ok) {
          setDataStatus((await statusResponse.json()) as DataStatusResponse)
        }
      } catch {
        /* ignore */
      }
    } catch (error) {
      setEmbeddingDeleteError(error instanceof Error ? error.message : 'Failed to delete embeddings.')
    } finally {
      setIsDeletingEmbeddings(false)
    }
  }

  const handleRunEvaluation = async (topK: number) => {
    setEvaluationError('')
    setEvaluationResult(null)
    setIsEvaluating(true)
    const k = Math.min(32, Math.max(1, topK))
    try {
      const parsedCases = JSON.parse(evalInput) as Array<{ query: string; expected_terms: string[] }>
      const response = await fetch(`${API_BASE_URL}/evaluation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_cases: parsedCases, top_k: k }),
      })
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const payload = (await response.json()) as EvaluationResponse
      setEvaluationResult(payload)
    } catch (error) {
      setEvaluationError(error instanceof Error ? error.message : 'Evaluation failed.')
    } finally {
      setIsEvaluating(false)
    }
  }

  return (
    <main className="app">
      <h1>Sitemap Group Ingestion</h1>

      <section className="card">
        <label htmlFor="sitemap-url" className="label">
          Sitemap URL
        </label>
        <div className="row">
          <input
            id="sitemap-url"
            className="input"
            type="url"
            value={sitemapUrl}
            onChange={(event) => setSitemapUrl(event.target.value)}
            placeholder="https://docs.example.com/sitemap.xml"
          />
          <button
            className="button"
            type="button"
            onClick={handleLoadSitemap}
            disabled={!sitemapUrl || isLoadingGroups}
          >
            {isLoadingGroups ? 'Loading...' : 'Load Sitemap'}
          </button>
        </div>
        {loadError ? <p className="error">{loadError}</p> : null}
      </section>

      {groups.length > 0 ? (
        <section className="card">
          <div className="groups-header">
            <h2>Groups</h2>
            <p>{groups.length} groups found</p>
          </div>
          {discoveryInfo ? (
            <div className="discovery-info">
              <p>
                Discovered <strong>{discoveryInfo.discoveredSitemapsCount}</strong> sitemaps and{' '}
                <strong>{discoveryInfo.totalUrls}</strong> URLs before crawl.
              </p>
              {discoveryInfo.sampleSitemaps.length > 0 ? (
                <details>
                  <summary>Show sample discovered sitemaps</summary>
                  <ul>
                    {discoveryInfo.sampleSitemaps.map((sitemap) => (
                      <li key={sitemap}>{sitemap}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </div>
          ) : null}
          <div className="row controls-row">
            <input
              className="input"
              type="text"
              value={groupSearch}
              onChange={(event) => setGroupSearch(event.target.value)}
              placeholder="Filter groups by name"
            />
            <button className="button secondary" type="button" onClick={handleSelectAllVisible}>
              Select Visible
            </button>
            <button className="button secondary" type="button" onClick={handleClearSelection}>
              Clear
            </button>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={skipScrapeIfExists}
              onChange={(event) => setSkipScrapeIfExists(event.target.checked)}
            />
            <span>Skip scrape if this URL was already extracted earlier</span>
          </label>
          <div className="groups-list">
            {visibleGroups.map((group) => (
              <label className="group-item" key={group.group_id}>
                <input
                  type="checkbox"
                  checked={selectedGroupIds.includes(group.group_id)}
                  onChange={() => handleToggleGroup(group.group_id)}
                />
                <div>
                  <p className="group-title">
                    {group.group_name} ({group.url_count})
                  </p>
                  <p className="group-meta">
                    Appears in {group.source_sitemaps_count} sitemap file(s)
                  </p>
                  <ul>
                    {group.sample_urls.map((sampleUrl) => (
                      <li key={sampleUrl}>{sampleUrl}</li>
                    ))}
                  </ul>
                </div>
              </label>
            ))}
          </div>
          <div className="row">
            <button
              className="button"
              type="button"
              disabled={selectedCount === 0 || isIngestingGroups}
              onClick={handleIngestGroups}
            >
              {isIngestingGroups ? 'Ingesting...' : `Ingest ${selectedCount} Groups`}
            </button>
          </div>
          {ingestError ? <p className="error">{ingestError}</p> : null}
        </section>
      ) : null}

      {ingestResult ? (
        <section className="card">
          <h2>Ingestion Result</h2>
          <p>Ingested pages: {ingestResult.ingested_count}</p>
          <p>Skipped pages: {ingestResult.skipped_count ?? 0}</p>
          <p>Failed pages: {ingestResult.failed_count ?? 0}</p>
          <ul>
            {ingestResult.items.slice(0, 10).map((item) => (
              <li key={item.source_url}>
                <strong>{item.title || '(No title)'}</strong> - {item.source_url}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="card">
        <h2>Chat with Extracted Knowledge</h2>
        <div className="row">
          <input
            className="input"
            type="text"
            value={chatQuestion}
            onChange={(event) => setChatQuestion(event.target.value)}
            placeholder="Ask a question about scraped content"
          />
          <button
            className="button"
            type="button"
            onClick={handleAskQuestion}
            disabled={!chatQuestion.trim() || isChatting}
          >
            {isChatting ? 'Generating...' : 'Ask'}
          </button>
        </div>
        {chatError ? <p className="error">{chatError}</p> : null}
        {chatAnswer ? (
          <div className="chat-answer">
            <p>{chatAnswer}</p>
            {chatSources.length > 0 ? (
              <>
                <p className="group-meta">Sources used:</p>
                <ul>
                  {chatSources.map((source) => (
                    <li key={source}>{source}</li>
                  ))}
                </ul>
              </>
            ) : null}
          </div>
        ) : null}

        <div className="stream-panel">
          <h3>Streaming chat (POST /chat/stream)</h3>
          <p className="group-meta">
            Uses the same question as above. Tokens appear in the box as LM Studio streams them (after
            retrieval). Cancel stops the HTTP request.
          </p>
          <div className="row">
            <button
              className="button"
              type="button"
              onClick={() => void handleAskStream()}
              disabled={!chatQuestion.trim() || isStreaming}
            >
              {isStreaming ? 'Streaming…' : 'Ask (stream)'}
            </button>
            <button
              className="button secondary"
              type="button"
              onClick={handleCancelStream}
              disabled={!isStreaming}
            >
              Cancel stream
            </button>
          </div>
          <textarea
            className="stream-output"
            readOnly
            value={streamAnswer}
            placeholder="Streamed assistant reply appears here…"
            aria-label="Streamed chat response"
          />
          {streamStatus ? <p className="stream-status">{streamStatus}</p> : null}
          {streamError ? <p className="error">{streamError}</p> : null}
          {streamMeta && streamMeta.sources.length > 0 ? (
            <div className="chat-answer" style={{ marginTop: '0.75rem' }}>
              <p className="group-meta">Sources (from stream meta):</p>
              <ul>
                {streamMeta.sources.map((source) => (
                  <li key={source}>{source}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </section>

      <section className="card">
        <h2>Data Cleanup</h2>
        <div className="row">
          <button className="button secondary" type="button" onClick={handleLoadDataStatus}>
            Refresh Data Status
          </button>
        </div>
        {dataStatus ? (
          <div className="data-status">
            <p>Raw files: {dataStatus.raw_count}</p>
            <p>Clean files: {dataStatus.clean_count}</p>
            <p>Processing files: {dataStatus.processing_count}</p>
            {dataStatus.enrichment_count != null ? (
              <p>Enrichment files: {dataStatus.enrichment_count}</p>
            ) : null}
            {dataStatus.chunks_count != null ? <p>Chunk manifests: {dataStatus.chunks_count}</p> : null}
            {dataStatus.qdrant_collection ? (
              <p>
                Qdrant collection <code className="inline-code">{dataStatus.qdrant_collection}</code>
                {dataStatus.qdrant_points_count != null && dataStatus.qdrant_points_count !== undefined
                  ? ` — points: ${dataStatus.qdrant_points_count}`
                  : null}
                {dataStatus.qdrant_error ? (
                  <span className="error-inline"> ({dataStatus.qdrant_error})</span>
                ) : null}
              </p>
            ) : null}
            <p className="group-meta">Raw dir: {dataStatus.raw_dir}</p>
            <p className="group-meta">Clean dir: {dataStatus.clean_dir}</p>
            <p className="group-meta">Processing dir: {dataStatus.processing_dir}</p>
            {dataStatus.enriched_dir ? (
              <p className="group-meta">Enriched dir: {dataStatus.enriched_dir}</p>
            ) : null}
            {dataStatus.semantic_chunks_dir ? (
              <p className="group-meta">Semantic chunks dir: {dataStatus.semantic_chunks_dir}</p>
            ) : null}
          </div>
        ) : null}
        <div className="row wrap-row">
          <button
            className="button"
            type="button"
            disabled={isRunningPipelineStep}
            onClick={() => handleRunPipelineStep('run-clean')}
          >
            Run Clean
          </button>
          <button
            className="button"
            type="button"
            disabled={isRunningPipelineStep}
            onClick={() => handleRunPipelineStep('run-processing')}
          >
            Run Processing
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(true, false, false, false, false, 'Clear Raw')}
          >
            Clear Raw
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(false, true, false, false, false, 'Clear Clean')}
          >
            Clear Clean
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(false, false, true, false, false, 'Clear Processing')}
          >
            Clear Processing
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(false, false, false, true, false, 'Clear Enriched')}
          >
            Clear Enriched
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(false, false, false, false, true, 'Clear semantic chunks')}
          >
            Clear semantic chunks
          </button>
          <button
            className="button danger"
            type="button"
            disabled={isCleaningData || isRunningPipelineStep}
            onClick={() => handleCleanupData(true, true, true, true, true, 'Clear All Data')}
          >
            Clear All
          </button>
        </div>
        {cleanupMessage ? <p>{cleanupMessage}</p> : null}
        {cleanupError ? <p className="error">{cleanupError}</p> : null}
      </section>

      <section className="card">
        <h2>Step 6 — Enrichment</h2>
        <p className="group-meta">
          Runs the LLM over each <code className="inline-code">*.md</code> in processing (needs matching{' '}
          <code className="inline-code">*.metadata.json</code>). Artifacts are written to{' '}
          <code className="inline-code">backend/data/enriched</code> as{' '}
          <code className="inline-code">{'{stem}.enrichment.json'}</code> (same stem as the markdown file).
        </p>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={enrichmentForce}
            onChange={(event) => setEnrichmentForce(event.target.checked)}
            disabled={isEnriching}
          />
          <span>Overwrite existing enrichment files</span>
        </label>
        <div className="row wrap-row">
          <button className="button" type="button" disabled={isEnriching} onClick={() => void handleRunEnrichment()}>
            {isEnriching ? 'Enriching…' : 'Run enrichment'}
          </button>
        </div>
        {enrichmentMessage ? <p>{enrichmentMessage}</p> : null}
        {enrichmentError ? <p className="error">{enrichmentError}</p> : null}
      </section>

      <section className="card">
        <h2>Step 7 — Semantic chunking</h2>
        <p className="group-meta">
          Writes one <code className="inline-code">.chunks.json</code> per markdown into{' '}
          <code className="inline-code">backend/data/semantic-chunks</code> (same stem as the{' '}
          <code className="inline-code">.md</code> in processing). Reads markdown from processing; loads Step 6
          relationships from the enriched directory when present.
          Live progress uses a server push stream (SSE), not browser polling.
        </p>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={chunkingForce}
            onChange={(event) => setChunkingForce(event.target.checked)}
            disabled={chunkingJobId !== null}
          />
          <span>Overwrite existing chunk manifests</span>
        </label>
        <div className="row wrap-row">
          <button
            className="button"
            type="button"
            disabled={chunkingJobId !== null}
            onClick={() => void handleStartChunking()}
          >
            {chunkingJobId !== null ? 'Chunking…' : 'Run semantic chunking'}
          </button>
        </div>
        {chunkingJob ? (
          <div className="progress-block">
            <div className="progress-track" role="progressbar" aria-valuenow={chunkingJob.percent} aria-valuemin={0} aria-valuemax={100}>
              <div className="progress-fill" style={{ width: `${chunkingJob.percent}%` }} />
            </div>
            <p className="group-meta progress-caption">
              {chunkingJob.status === 'running'
                ? `Processing ${chunkingJob.processed_files} / ${chunkingJob.total_files} (${chunkingJob.percent}%)${chunkingJob.current_file ? ` — ${chunkingJob.current_file}` : ''}`
                : null}
              {chunkingJob.status === 'completed' && chunkingJob.result ? (
                <>
                  Done — wrote {chunkingJob.result.files_written}, skipped {chunkingJob.result.files_skipped},
                  failed {chunkingJob.result.files_failed}, total chunks indexed in manifests:{' '}
                  {chunkingJob.result.chunks_total}.
                </>
              ) : null}
              {chunkingJob.status === 'failed' && chunkingJob.error ? (
                <span className="error-inline">{chunkingJob.error}</span>
              ) : null}
              {chunkingJob.status === 'unknown' && chunkingJob.error ? (
                <span className="error-inline">{chunkingJob.error}</span>
              ) : null}
            </p>
          </div>
        ) : null}
        {chunkingError ? <p className="error">{chunkingError}</p> : null}
      </section>

      <section className="card">
        <h2>Step 8 — Embeddings + Qdrant</h2>
        <p className="group-meta">
          Pipeline: (1) download / verify model weights in the Hugging Face cache, (2) load the model into
          memory, (3) ensure the Qdrant collection, (4) embed <code className="inline-code">*.chunks.json</code>{' '}
          from <code className="inline-code">backend/data/semantic-chunks</code> and upsert. Model id comes from{' '}
          <code className="inline-code">EMBEDDING_MODEL_NAME</code> (default BGE-M3, 1024-dim). Chat still uses
          keyword retrieval until Step 9.
        </p>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={indexingRecreateCollection}
            onChange={(event) => setIndexingRecreateCollection(event.target.checked)}
            disabled={indexingJobId !== null}
          />
          <span>Recreate Qdrant collection (deletes existing vectors in this collection)</span>
        </label>
        <div className="row wrap-row">
          <button
            className="button"
            type="button"
            disabled={indexingJobId !== null || isDeletingEmbeddings}
            onClick={() => void handleStartIndexing()}
          >
            {indexingJobId !== null ? 'Persisting…' : 'Persist Vector Embeddings'}
          </button>
          <button
            className="button danger"
            type="button"
            disabled={indexingJobId !== null || isDeletingEmbeddings}
            onClick={() => void handleDeleteEmbeddings()}
          >
            {isDeletingEmbeddings ? 'Deleting…' : 'Delete Embeddings'}
          </button>
        </div>
        {embeddingDeleteMessage ? <p>{embeddingDeleteMessage}</p> : null}
        {embeddingDeleteError ? <p className="error">{embeddingDeleteError}</p> : null}
        {indexingJob ? (
          <div className="progress-block">
            <div
              className="progress-track"
              role="progressbar"
              aria-valuenow={indexingJob.percent}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div className="progress-fill" style={{ width: `${indexingJob.percent}%` }} />
            </div>
            <p className="group-meta progress-caption">
              {indexingJob.status === 'running' ? (
                <>
                  {indexingJob.phase === 'queued' ? <span>Queued — starting worker…</span> : null}
                  {indexingJob.phase === 'downloading_model' ? (
                    <span>
                      Step 1/4 — Downloading / verifying model weights on disk (Hugging Face cache, resumes if
                      interrupted). Then memory load, then Qdrant, then upsert. Heartbeat in API logs every 45s:{' '}
                      <code className="inline-code">Still downloading model files</code>.
                      {indexingJob.current_file ? (
                        <>
                          {' '}
                          Repo: <code className="inline-code">{indexingJob.current_file}</code>
                        </>
                      ) : null}
                    </span>
                  ) : null}
                  {indexingJob.phase === 'loading_model' ? (
                    <span>
                      Step 2/4 — Loading model into RAM (after files are on disk). Large models can take several
                      minutes. Logs: <code className="inline-code">Still loading model into memory</code>.
                      {indexingJob.current_file ? (
                        <>
                          {' '}
                          <code className="inline-code">{indexingJob.current_file}</code>
                        </>
                      ) : null}
                    </span>
                  ) : null}
                  {indexingJob.phase === 'qdrant' ? <span>Step 3/4 — Preparing Qdrant collection…</span> : null}
                  {indexingJob.phase === 'embedding' ||
                  (indexingJob.phase == null && indexingJob.status === 'running') ? (
                    <span>
                      Step 4/4 — Persisting manifests {indexingJob.processed_files} / {indexingJob.total_files} (
                      {indexingJob.percent}%)
                      {indexingJob.current_file ? ` — ${indexingJob.current_file}` : ''}
                      {indexingJob.points_indexed != null && indexingJob.points_indexed > 0 ? (
                        <> · Qdrant points so far: {indexingJob.points_indexed}</>
                      ) : null}
                    </span>
                  ) : null}
                </>
              ) : null}
              {indexingJob.status === 'completed' && indexingJob.result ? (
                <>
                  Done — processed {indexingJob.result.manifests_processed}, failed manifests{' '}
                  {indexingJob.result.manifests_failed}, points upserted {indexingJob.result.points_upserted}.
                </>
              ) : null}
              {indexingJob.status === 'failed' && indexingJob.error ? (
                <span className="error-inline">{indexingJob.error}</span>
              ) : null}
              {indexingJob.status === 'unknown' && indexingJob.error ? (
                <span className="error-inline">{indexingJob.error}</span>
              ) : null}
            </p>
          </div>
        ) : null}
        {indexingError ? <p className="error">{indexingError}</p> : null}
      </section>

      <section className="card">
        <h2>Evaluation</h2>
        <p className="group-meta">
          JSON array of <code className="inline-code">query</code> and{' '}
          <code className="inline-code">expected_terms</code>. Runs the same pipeline as Chat: when Qdrant has
          points, retrieval is dense + hybrid + rerank; otherwise keyword manifests / markdown fallback.
        </p>
        <textarea
          className="eval-textarea"
          value={evalInput}
          onChange={(event) => setEvalInput(event.target.value)}
        />
        <div className="row wrap-row eval-actions" style={{ alignItems: 'center', gap: '0.75rem' }}>
          <label className="group-meta" htmlFor="eval-top-k" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            top_k
            <input
              id="eval-top-k"
              className="input"
              type="number"
              min={1}
              max={32}
              value={evalTopK}
              onChange={(e) => setEvalTopK(Number.parseInt(e.target.value, 10) || 4)}
              style={{ width: '4rem' }}
            />
          </label>
          <button
            className="button"
            type="button"
            onClick={() => void handleRunEvaluation(evalTopK)}
            disabled={isEvaluating}
          >
            {isEvaluating ? 'Running...' : 'Run vector evaluation'}
          </button>
          <button
            className="button secondary"
            type="button"
            disabled
            title="Reserved for graph-native retrieval when RagGraph is implemented."
          >
            RagGraph evaluation
          </button>
        </div>
        {evaluationError ? <p className="error">{evaluationError}</p> : null}
        {evaluationResult ? (
          <div className="data-status">
            <p className="group-meta">
              Hallucination estimate = 1 − grounded hit rate. A case is &quot;grounded&quot; only if the
              model answer contains at least one expected term (substring, case-insensitive). It is not a
              full faithfulness check against sources.
            </p>
            <p>Cases: {evaluationResult.cases_count}</p>
            <p>Retrieval hit rate: {evaluationResult.retrieval_hit_rate}</p>
            <p>Grounded hit rate: {evaluationResult.grounded_hit_rate}</p>
            <p>Hallucination estimate: {evaluationResult.hallucination_rate_estimate}</p>
            <details className="eval-cases">
              <summary>Per-case breakdown (see why a score is 0 or 1)</summary>
              <ul className="eval-case-list">
                {evaluationResult.cases.map((evaluationCase, index) => (
                  <li key={`${evaluationCase.query}-${index}`}>
                    <p>
                      <strong>Q:</strong> {evaluationCase.query}
                    </p>
                    <p className="group-meta">
                      Expected: {evaluationCase.expected_terms.join(', ')} · Retrieved chunks:{' '}
                      {evaluationCase.retrieved_count}
                      {evaluationCase.retrieval_mode ? ` · mode: ${evaluationCase.retrieval_mode}` : ''} ·
                      Grounded: {evaluationCase.grounded_hit}
                      {evaluationCase.matched_term ? ` (matched: &quot;${evaluationCase.matched_term}&quot;)` : ' (no term matched)'}
                    </p>
                    {evaluationCase.chunk_ids && evaluationCase.chunk_ids.length > 0 ? (
                      <p className="group-meta eval-chunk-ids">
                        chunk_ids:{' '}
                        <code className="inline-code">
                          {evaluationCase.chunk_ids.slice(0, 4).join(', ')}
                          {evaluationCase.chunk_ids.length > 4 ? '…' : ''}
                        </code>
                      </p>
                    ) : null}
                    <p className="eval-answer-preview">{evaluationCase.answer_preview || '(empty answer)'}</p>
                  </li>
                ))}
              </ul>
            </details>
          </div>
        ) : null}
      </section>
    </main>
  )
}

export default App
