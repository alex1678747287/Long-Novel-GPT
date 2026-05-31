import { defineStore } from 'pinia'
import { api } from '../api/v2'

// Tiny debounce helper. We don't want every keystroke to PATCH the chapter,
// but we also don't want to lose the last edit on tab close. So: trail-edge
// debounce ~600ms.
function debounce(fn, ms) {
  let t = null
  let pending = null
  const wrapper = (...args) => {
    pending = args
    if (t) clearTimeout(t)
    t = setTimeout(() => {
      t = null
      const p = pending
      pending = null
      fn(...p)
    }, ms)
  }
  wrapper.flush = () => {
    if (t) {
      clearTimeout(t)
      t = null
      const p = pending
      pending = null
      if (p) fn(...p)
    }
  }
  return wrapper
}

// Holds the imported novel + chapters + per-chapter rewrite state. Backed by
// SQLite via /api/v2/novels — so refreshing the page or restarting the
// container preserves everything.
export const useNovelStore = defineStore('novel', {
  state: () => ({
    novels: [],            // [{id, title, chapter_count, done_count, analysis_status, updated_at}]
    activeNovelId: localStorage.getItem('active_novel_id') || null,
    title: '',
    genre: '',
    target_genre: '',
    style_tone: '',
    rewrite_strength: '',
    chapters: [],          // {id, title, summary, content, rewritten, overlap, quality_score, status}
    activeChapterId: null,
    splitting: false,
    splitMode: '',
    loading: false,
    saving: false,         // shown in the UI to indicate "auto-saving..."
    analysisStatus: 'idle', // 'idle' | 'running' | 'done' | 'error'
    _analysisPoll: null,
  }),
  getters: {
    activeChapter(state) {
      return state.chapters.find((c) => c.id === state.activeChapterId) || null
    },
    activeNovel(state) {
      return state.novels.find((n) => n.id === state.activeNovelId) || null
    },
  },
  actions: {
    /** Load the list of novels and auto-restore the most recent one (or the
     *  one whose id is in localStorage). Call this once on app startup. */
    async loadAll() {
      this.loading = true
      try {
        await this.refreshNovelList()
        // Pick which novel to open: stored id if still valid, else most recent.
        let pickId = this.activeNovelId
        if (!this.novels.find((n) => n.id === pickId)) {
          pickId = this.novels[0]?.id || null
        }
        if (pickId) {
          await this.openNovel(pickId)
        } else {
          this._clearLocal()
        }
      } finally {
        this.loading = false
      }
    },

    async refreshNovelList() {
      this.novels = await api.listNovels()
      return this.novels
    },

    async openNovel(id) {
      this.loading = true
      try {
        const novel = await api.getNovel(id)
        this.activeNovelId = id
        localStorage.setItem('active_novel_id', id)
        this.title = novel.title
        this.genre = novel.genre || ''
        this.target_genre = novel.target_genre || ''
        this.style_tone = novel.style_tone || ''
        this.rewrite_strength = novel.rewrite_strength || ''
        this.splitMode = novel.split_mode || ''
        this.analysisStatus = novel.analysis_status || 'idle'
        this.chapters = (novel.chapters || []).map((c) => ({
          id: c.id,
          title: c.title,
          summary: c.summary || '',
          content: c.content || '',
          rewritten: c.rewritten || '',
          rewritten_script: c.rewritten_script || '',
          script_status: c.script_status || (c.rewritten_script ? 'done' : 'idle'),
          overlap: c.overlap == null ? null : Number(c.overlap),
          quality_score: c.quality_score == null ? null : Number(c.quality_score),
          quality_grade: c.quality_grade || '',
          quality_issues: c.quality_issues || '',
          status: c.status || 'idle',
        }))
        this.activeChapterId = this.chapters[0]?.id || null
        // Backfill: legacy novels created before the analyzer existed have
        // analysis_status='idle' even though they have content. Auto-kick a
        // one-time analysis so cross-chapter consistency works on them too.
        if (this.analysisStatus === 'idle' && this.chapters.length > 0) {
          this._maybeBackfillAnalysis(id)
        } else {
          this._pollAnalysisStatus()
        }
      } finally {
        this.loading = false
      }
    },

    async _maybeBackfillAnalysis(id) {
      try {
        await api.reanalyzeNovel(id)
        this.analysisStatus = 'running'
        this._pollAnalysisStatus()
      } catch (e) {
        console.warn('analysis backfill failed', e)
      }
    },

    /** Poll the analysis status while it's 'running'. Stops on done/error/idle. */
    _pollAnalysisStatus() {
      if (this._analysisPoll) {
        clearTimeout(this._analysisPoll)
        this._analysisPoll = null
      }
      if (this.analysisStatus !== 'running') return
      const tick = async () => {
        try {
          const list = await api.listNovels()
          this.novels = list
          const cur = list.find((n) => n.id === this.activeNovelId)
          if (cur) {
            this.analysisStatus = cur.analysis_status || 'idle'
          }
        } catch (e) {
          // Keep polling — transient network errors shouldn't kill the loop,
          // but surface them in devtools so we can spot a broken backend.
          console.warn('analysis poll failed', e)
        }
        if (this.analysisStatus === 'running') {
          this._analysisPoll = setTimeout(tick, 3000)
        } else {
          this._analysisPoll = null
        }
      }
      this._analysisPoll = setTimeout(tick, 2000)
    },

    /** Create a brand-new novel from raw pasted text. Tries a local fast-split;
     *  if that fails (no explicit chapter headers), stores as a single chapter
     *  so the user can hit 拆章 to call the LLM splitter.
     *  @param customTitle optional. If omitted, derive from the first non-blank line. */
    async createNovelFromText(rawText, customTitle = '', meta = {}) {
      const title =
        (customTitle || '').trim() ||
        (rawText.split('\n').find((l) => l.trim()) || '未命名').slice(0, 30)
      // Fast local split first — saves a model call when chapter headers exist.
      let chapters = []
      let splitMode = ''
      try {
        const res = await api.split({
          text: rawText,
          max_chapter_size: meta.max_chapter_size,
        })
        chapters = res.chapters.map((c, i) => ({
          title: c.title || `第${i + 1}章`,
          summary: c.summary || '',
          content: c.content || '',
        }))
        splitMode = res.mode
      } catch (e) {
        // Local regex couldn't find chapter headers AND the model isn't
        // configured (or the call failed). Fall back to a single-chapter
        // novel — user can still rewrite. Log so devtools shows why.
        console.warn('chapter split failed, falling back to single chapter', e)
      }
      if (!chapters.length) {
        chapters = [{ title, summary: '', content: rawText }]
      }
      const novel = await api.createNovel({
        title,
        genre: meta.genre || '',
        target_genre: meta.target_genre || '',
        style_tone: meta.style_tone || '',
        rewrite_strength: meta.rewrite_strength || '',
        chapters,
        split_mode: splitMode,
      })
      await this.refreshNovelList()
      await this.openNovel(novel.id)
      // Backend kicked off analysis in the background. Surface that to the
      // UI even before the poll catches up.
      this.analysisStatus = 'running'
      this._pollAnalysisStatus()
      return novel
    },

    /** Re-split the currently open novel using the LLM splitter. Wipes and
     *  re-inserts chapters in the DB. */
    async resplitActiveNovel(options = {}) {
      if (!this.activeNovelId) return
      this.splitting = true
      try {
        // Recompose the full text from existing chapters and resplit.
        const fullText = this.chapters
          .map((c) => `${c.title}\n${c.content}`)
          .join('\n\n')
        const res = await api.split({
          text: fullText,
          max_chapter_size: options.max_chapter_size,
        })
        await api.replaceChapters(this.activeNovelId, {
          chapters: res.chapters,
          split_mode: res.mode,
          max_chapter_size: options.max_chapter_size,
        })
        await this.openNovel(this.activeNovelId)
        this.splitMode = res.mode
        // Surface a soft warning when the backend had to fall back
        return { chapters: this.chapters, mode: res.mode, warning: res.warning }
      } finally {
        this.splitting = false
      }
    },

    async deleteNovel(id) {
      await api.deleteNovel(id)
      this.novels = this.novels.filter((n) => n.id !== id)
      if (this.activeNovelId === id) {
        const next = this.novels[0]
        if (next) {
          await this.openNovel(next.id)
        } else {
          this._clearLocal()
        }
      }
    },

    /** Rename the active novel. */
    async renameActiveNovel(title) {
      if (!this.activeNovelId) return
      const updated = await api.patchNovel(this.activeNovelId, { title })
      this.title = updated.title
      this.genre = updated.genre || this.genre
      this.target_genre = updated.target_genre || this.target_genre
      this.style_tone = updated.style_tone || this.style_tone
      this.rewrite_strength = updated.rewrite_strength || this.rewrite_strength
      await this.refreshNovelList()
    },

    setActive(id) {
      this.activeChapterId = id
    },

    /** Update a chapter's content. Debounced PATCH to backend (~600ms). */
    setChapterContent(id, content) {
      const c = this.chapters.find((x) => x.id === id)
      if (!c) return
      c.content = content
      this._debouncedPatchChapter(id, { content })
    },

    /** Find the script-builtin prompt id (assumes it exists in the prompts list).
     *  Used by the "转剧本" chapter-level button. */
    getScriptPromptId(promptStore) {
      const p = (promptStore?.prompts || []).find(
        (x) => x.is_builtin && ['转剧本', '洗稿剧本版', '精修剧本版'].includes(x.name)
      )
      return p?.id || null
    },

    /** Rename a chapter inline. Persists immediately (no debounce — titles
     *  change one-shot, not key-by-key). */
    async renameChapter(id, title) {
      const c = this.chapters.find((x) => x.id === id)
      if (!c) return
      const trimmed = (title || '').trim()
      if (!trimmed || trimmed === c.title) return
      c.title = trimmed
      try {
        await api.patchChapter(id, { title: trimmed })
      } catch (e) {
        console.warn('rename chapter failed', e)
      }
    },

    /** Called both during streaming (frequent, in-memory only) and at done-time
     *  (one PATCH). Pass persist=true to write to DB.
     *  variant: 'base' (writes to rewritten) | 'script' (writes to rewritten_script). */
    setRewritten(id, rewritten, overlap = undefined, status = 'idle', persist = false, variant = 'base', quality = undefined, refreshList = true) {
      const c = this.chapters.find((x) => x.id === id)
      if (!c) return Promise.resolve(false)
      const field = variant === 'script' ? 'rewritten_script' : 'rewritten'
      c[field] = rewritten
      if (overlap !== undefined) c.overlap = overlap
      if (variant === 'base' && quality !== undefined) {
        c.quality_score = quality ? Number(quality.score) : null
        c.quality_grade = quality?.delivery_status || quality?.delivery_label || quality?.grade || ''
        c.quality_issues = quality?.issues ? JSON.stringify(quality.issues) : ''
      }
      if (status !== null) {
        if (variant === 'script') c.script_status = status
        else c.status = status
      }
      if (!persist) {
        return Promise.resolve(true)
      }

      const payload = { [field]: rewritten }
      if (overlap !== undefined) payload.overlap = overlap
      if (variant === 'base' && quality !== undefined) {
        payload.quality_score = quality ? Number(quality.score) : null
        payload.quality_grade = quality?.delivery_status || quality?.delivery_label || quality?.grade || ''
        payload.quality_issues = quality?.issues ? JSON.stringify(quality.issues) : ''
      }
      if (status !== null) {
        if (variant === 'script') payload.script_status = status
        else payload.status = status
      }
      this.saving = true
      return api
        .patchChapter(id, payload)
        .then(async () => {
          this.saving = false
          if (refreshList) await this.refreshNovelList()
          return true
        })
        .catch(() => {
          this.saving = false
          return false
        })
    },

    // Internal debounced PATCH per chapter. Each chapter gets its own
    // debouncer so edits on different chapters don't trample each other.
    _debouncedPatchChapter(id, fields) {
      if (!this._patchers) this._patchers = new Map()
      let fn = this._patchers.get(id)
      if (!fn) {
        fn = debounce((payload) => {
          this.saving = true
          api
            .patchChapter(id, payload)
            .catch(() => {})
            .finally(() => (this.saving = false))
        }, 600)
        this._patchers.set(id, fn)
      }
      fn(fields)
    },

    /** Flush all pending edits (called e.g. on page-unload). */
    flushPendingEdits() {
      if (!this._patchers) return
      for (const fn of this._patchers.values()) fn.flush()
    },

    _clearLocal() {
      this.activeNovelId = null
      localStorage.removeItem('active_novel_id')
      this.title = ''
      this.genre = ''
      this.target_genre = ''
      this.style_tone = ''
      this.rewrite_strength = ''
      this.chapters = []
      this.activeChapterId = null
      this.splitMode = ''
    },
  },
})
