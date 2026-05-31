// Tiny fetch wrapper for the /api/v2/* endpoints exposed by backend/v2/api.py.
// All calls go through the nginx /api proxy in production, or through the
// Vite dev server proxy in development — see vite.config.js.

const BASE = '/api/v2'
const IDLE_TIMEOUT_MS = 300000

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    // Read body as text first — Response.body is a single-shot stream, so
    // we can't fall back to res.text() if res.json() fails (that throws
    // "body stream already read"). Parse the text manually.
    let detail = ''
    try {
      const raw = await res.text()
      try {
        const body = JSON.parse(raw)
        detail = body.error || body.message || JSON.stringify(body)
      } catch {
        detail = raw
      }
    } catch (e) {
      detail = `(无法读取响应: ${e.message})`
    }
    throw new Error(`${res.status} ${detail}`)
  }
  return res.json()
}

export const api = {
  presets: () => request('/presets'),

  listModels: () => request('/models'),
  upsertModel: (payload) =>
    request('/models', { method: 'POST', body: JSON.stringify(payload) }),
  deleteModel: (id) => request(`/models/${id}`, { method: 'DELETE' }),
  activateModel: (id) =>
    request(`/models/${id}/activate`, { method: 'POST', body: '{}' }),
  testModel: (id) =>
    request(`/models/${id}/test`, { method: 'POST', body: '{}' }),

  listPrompts: () => request('/prompts'),
  upsertPrompt: (payload) =>
    request('/prompts', { method: 'POST', body: JSON.stringify(payload) }),
  deletePrompt: (id) => request(`/prompts/${id}`, { method: 'DELETE' }),

  getSystem: () => request('/system'),
  setSystem: (payload) =>
    request('/system', { method: 'POST', body: JSON.stringify(payload) }),

  split: (payload) =>
    request('/split', { method: 'POST', body: JSON.stringify(payload) }),

  // Persistent novel/chapter storage (SQLite-backed).
  listNovels: () => request('/novels'),
  getNovel: (id) => request(`/novels/${id}`),
  createNovel: (payload) =>
    request('/novels', { method: 'POST', body: JSON.stringify(payload) }),
  patchNovel: (id, payload) =>
    request(`/novels/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  replaceChapters: (id, payload) =>
    request(`/novels/${id}/chapters`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteNovel: (id) => request(`/novels/${id}`, { method: 'DELETE' }),
  patchChapter: (id, payload) =>
    request(`/chapters/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  reanalyzeNovel: (id) =>
    request(`/novels/${id}/analyze`, { method: 'POST', body: '{}' }),

  // Upload a .docx file and return the extracted plain text. Uses a raw
  // fetch because the helper above only handles JSON bodies.
  parseDocx: async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${BASE}/parse_docx`, { method: 'POST', body: fd })
    if (!res.ok) {
      let detail = ''
      try {
        const raw = await res.text()
        try {
          const body = JSON.parse(raw)
          detail = body.error || JSON.stringify(body)
        } catch { detail = raw }
      } catch (e) { detail = `(无法读取响应: ${e.message})` }
      throw new Error(`${res.status} ${detail}`)
    }
    return res.json()
  },

  parseZipCorpus: async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${BASE}/eval/import_zip`, { method: 'POST', body: fd })
    if (!res.ok) {
      let detail = ''
      try {
        const raw = await res.text()
        try {
          const body = JSON.parse(raw)
          detail = body.error || JSON.stringify(body)
        } catch { detail = raw }
      } catch (e) { detail = `(无法读取响应: ${e.message})` }
      throw new Error(`${res.status} ${detail}`)
    }
    return res.json()
  },

  qualityScore: (payload) =>
    request('/quality/score', { method: 'POST', body: JSON.stringify(payload) }),

  enqueueChapterRewrite: (chapterId, payload) =>
    request(`/chapters/${chapterId}/rewrite-jobs`, {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
  enqueueNovelRewrite: (novelId, payload) =>
    request(`/novels/${novelId}/rewrite-jobs`, {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
  getRewriteJob: (jobId) => request(`/rewrite-jobs/${jobId}`),
  listNovelRewriteJobs: (novelId, options = {}) => {
    const params = new URLSearchParams()
    if (options.batch_id) params.set('batch_id', options.batch_id)
    if (options.active) params.set('active', 'true')
    const suffix = params.toString() ? `?${params.toString()}` : ''
    return request(`/novels/${novelId}/rewrite-jobs${suffix}`)
  },
  cancelRewriteJob: (jobId) =>
    request(`/rewrite-jobs/${jobId}/cancel`, { method: 'POST', body: '{}' }),

  exportBackup: () => request('/backup/export'),
  importBackup: (data, merge = true) =>
    request('/backup/import', {
      method: 'POST',
      body: JSON.stringify({ data, merge }),
    }),
}

// Server-Sent-Events helper for /api/v2/rewrite. The caller passes onChunk
// and onDone callbacks. Returns an AbortController so the caller can cancel.
export function streamRewrite(payload, { onChunk, onDone, onError, onAbort }) {
  const controller = new AbortController()
  let idleTimer = null
  let timedOut = false
  const clearIdleTimer = () => {
    if (idleTimer) {
      clearTimeout(idleTimer)
      idleTimer = null
    }
  }
  const resetIdleTimer = () => {
    clearIdleTimer()
    idleTimer = setTimeout(() => {
      timedOut = true
      controller.abort()
      onError?.(new Error('模型长时间没有返回内容，请重试'))
    }, IDLE_TIMEOUT_MS)
  }
  ;(async () => {
    try {
      resetIdleTimer()
      const res = await fetch(`${BASE}/rewrite`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })
      if (!res.ok || !res.body) {
        clearIdleTimer()
        const text = await res.text()
        onError?.(new Error(`${res.status} ${text}`))
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let sawDone = false
      // SSE: events separated by "\n\n". Each event has lines starting with
      // "data: ". We only emit on full events to avoid mid-JSON parsing.
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        resetIdleTimer()
        buffer += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffer.indexOf('\n\n')) >= 0) {
          const rawEvent = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          const dataLine = rawEvent
            .split('\n')
            .find((line) => line.startsWith('data:'))
          if (!dataLine) continue
          try {
            const parsed = JSON.parse(dataLine.slice(5).trim())
            if (parsed.error) {
              sawDone = true
              clearIdleTimer()
              onError?.(new Error(parsed.error))
              return
            } else if (parsed.done) {
              sawDone = true
              clearIdleTimer()
              onDone?.(parsed)
              return
            } else {
              onChunk?.(parsed)
            }
          } catch (e) {
            // ignore — partial chunk; keep buffering
          }
        }
      }
      decoder.decode()
      clearIdleTimer()
      if (!sawDone && !controller.signal.aborted) {
        onError?.(new Error('流式响应提前结束，请重试'))
      }
    } catch (e) {
      clearIdleTimer()
      if (e.name === 'AbortError') {
        if (!timedOut) onAbort?.()
      }
      else onError?.(e)
    }
  })()
  return controller
}

// 4-gram overlap ratio between two Chinese strings. Used to surface a
// "降重指标" in the workbench. Ignores ASCII whitespace.
export function overlap4gram(rewritten, original) {
  const normalize = (s) => s.replace(/\s+/g, '')
  const a = normalize(rewritten)
  const b = normalize(original)
  if (a.length < 4 || b.length < 4) return 0
  const aGrams = new Set()
  const bGrams = new Set()
  for (let i = 0; i + 4 <= a.length; i++) {
    aGrams.add(a.slice(i, i + 4))
  }
  for (let i = 0; i + 4 <= b.length; i++) {
    bGrams.add(b.slice(i, i + 4))
  }
  if (!aGrams.size || !bGrams.size) return 0
  let hits = 0
  for (const gram of aGrams) {
    if (bGrams.has(gram)) hits++
  }
  return hits / Math.min(aGrams.size, bGrams.size)
}
