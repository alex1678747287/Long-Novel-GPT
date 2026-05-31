<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api/v2'
import { useModelStore } from '../stores/models'
import { usePromptStore } from '../stores/prompts'
import { useNovelStore } from '../stores/novel'
import { useSystemStore } from '../stores/system'
import {
  Upload,
  MagicStick,
  Download,
  ArrowDown,
  CircleClose,
  DocumentCopy,
  EditPen,
  Plus,
  Document,
  Files,
  Right,
  Delete,
  Loading,
  Check,
  RefreshRight,
} from '@element-plus/icons-vue'

const modelStore = useModelStore()
const promptStore = usePromptStore()
const novelStore = useNovelStore()
const systemStore = useSystemStore()
const rewriteCapabilityMissingMessage = '改写能力未加载，请刷新页面或联系管理员'
const OVERLAP_EXCELLENT = 0.15
const OVERLAP_DELIVERABLE = 0.22
const DEFAULT_MAX_CHAPTER_SIZE = 2200
const ONE_MILLION_CONTEXT_MAX_CHAPTER_SIZE = 3000
const MID_CONTEXT_MAX_CHAPTER_SIZE = 2200
const REWRITE_JOB_POLL_MS = 2000
const REWRITE_ANALYSIS_WAIT_MS = 10 * 60 * 1000
const DEFAULT_REWRITE_QUALITY_MODE = 'auto'
const ACTIVE_REWRITE_CONTEXT_KEY = 'long-novel-gpt.active-rewrite-context'
const MAX_PARALLEL_REWRITE_NOVELS = 3

function isLargeContextModel(model) {
  if (!model) return false
  const explicitContext = Number(
    model.max_context_tokens ||
      model.context_window ||
      model.context_window_tokens ||
      model.context_length ||
      0
  )
  if (explicitContext >= 1000000) return true
  const name = String(model.model || model.name || '').toLowerCase().replaceAll('_', '-')
  const compact = name.replaceAll('-', '').replaceAll(' ', '')
  return [
    '1m',
    '1-million',
    '1000k',
    'claude-opus-4-7',
    'claude-opus-4-6',
    'claude-sonnet-4-7',
    'claude-sonnet-4-6',
    'gpt-5.5',
    'deepseek-v4',
    'deepseek-v4-pro',
    'deepseek-v4pro',
  ].some((pattern) => name.includes(pattern)) ||
    compact.includes('deepseekv4pro') ||
    compact.includes('gpt55')
}

function isMidContextModel(model) {
  if (!model) return false
  const explicitContext = Number(
    model.max_context_tokens ||
      model.context_window ||
      model.context_window_tokens ||
      model.context_length ||
      0
  )
  if (explicitContext >= 256000 && explicitContext < 1000000) return true
  const name = String(model.model || model.name || '').toLowerCase().replaceAll('_', '-')
  return name.includes('256k') || name.includes('512k')
}

function isDeepSeekModel(model) {
  if (!model) return false
  const name = String(model.model || model.name || '').toLowerCase().replaceAll('_', '-')
  return name.includes('deepseek')
}

// ---- import area ----
const importVisible = ref(false)
const importText = ref('')
const importTitle = ref('')
const fileInput = ref(null)
const evalCorpusSummary = ref(null)
const importSubmitting = ref(false)
const MAX_NOVEL_CHARS = 100000
const importCharCount = computed(() => importText.value.length)

function resetImportForm() {
  importTitle.value = ''
  importText.value = ''
  evalCorpusSummary.value = null
}

function openImport() {
  resetImportForm()
  importVisible.value = true
}

function pickFile() {
  fileInput.value?.click()
}

function onFileChange(e) {
  const file = e.target.files?.[0]
  if (!file) return
  loadTextFile(file)
  e.target.value = ''
}

const loadingDocx = ref(false)
const maxChapterSize = computed(() => {
  const configured = Number(systemStore.max_chapter_size) || DEFAULT_MAX_CHAPTER_SIZE
  if (isLargeContextModel(modelStore.activeModel)) {
    return ONE_MILLION_CONTEXT_MAX_CHAPTER_SIZE
  }
  if (isMidContextModel(modelStore.activeModel)) {
    return MID_CONTEXT_MAX_CHAPTER_SIZE
  }
  return Math.min(configured, DEFAULT_MAX_CHAPTER_SIZE)
})

async function loadTextFile(file) {
  if (!file) return
  const isDocx = /\.docx$/i.test(file.name)
  const isTxt = /\.txt$/i.test(file.name)
  const isZip = /\.zip$/i.test(file.name)
  if (!isDocx && !isTxt && !isZip) {
    ElMessage.warning('只支持 .txt / .docx / .zip 文件')
    return
  }
  // Auto-fill the title from the filename — user can still edit it.
  if (!importTitle.value.trim()) {
    importTitle.value = file.name.replace(/\.(txt|docx|zip)$/i, '').slice(0, 60)
  }
  if (isZip) {
    loadingDocx.value = true
    try {
      const res = await api.parseZipCorpus(file)
      evalCorpusSummary.value = res
      importText.value = ''
      ElMessage.success(`测试资料包已导入 · ${res.pair_count || 0} 组原稿/精修`)
    } catch (e) {
      ElMessage.error('导入测试资料包失败：' + e.message)
    } finally {
      loadingDocx.value = false
    }
    return
  }
  if (isDocx) {
    // Server-side parse: ship the .docx upstream, get back plain text.
    loadingDocx.value = true
    try {
      const res = await api.parseDocx(file)
      importText.value = res.text || ''
      if (importText.value.length > MAX_NOVEL_CHARS) {
        ElMessage.warning(`当前 ${importText.value.length} 字，超过 10w 字上限，请删减后再创建`)
      }
      ElMessage.success(`已解析 docx · ${res.paragraphs || 0} 段，共 ${importText.value.length} 字`)
    } catch (e) {
      ElMessage.error('解析 docx 失败：' + e.message)
    } finally {
      loadingDocx.value = false
    }
    return
  }
  // TXT path — local FileReader handles encoding fine for utf-8.
  try {
    const buffer = await file.arrayBuffer()
    let decoded = ''
    try {
      decoded = new TextDecoder('utf-8', { fatal: true }).decode(buffer)
    } catch {
      try {
        decoded = new TextDecoder('gb18030').decode(buffer)
      } catch {
        decoded = new TextDecoder('utf-8').decode(buffer)
      }
    }
    importText.value = decoded
  } catch (e) {
    ElMessage.error('读取 TXT 失败：' + e.message)
    return
  }
  if (importText.value.length > MAX_NOVEL_CHARS) {
    ElMessage.warning(`当前 ${importText.value.length} 字，超过 10w 字上限，请删减后再创建`)
  }
}

// Drag-and-drop into the import dialog.
const importDragOver = ref(false)
function onImportDragEnter(e) {
  e.preventDefault()
  importDragOver.value = true
}
function onImportDragLeave() {
  importDragOver.value = false
}
function onImportDrop(e) {
  e.preventDefault()
  importDragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) loadTextFile(file)
}

async function confirmImport() {
  if (importSubmitting.value) return
  if (!importText.value.trim()) {
    if (evalCorpusSummary.value) {
      importVisible.value = false
      resetImportForm()
      ElMessage.success(`测试资料包已保存为评测集 · ${evalCorpusSummary.value.pair_count || 0} 组`)
      return
    }
    ElMessage.warning('请粘贴或上传原稿')
    return
  }
  if (importText.value.length > MAX_NOVEL_CHARS) {
    ElMessage.warning(`当前 ${importText.value.length} 字，单次录入最多支持 10w 字以内`)
    return
  }
  const text = importText.value
  const title = importTitle.value.trim()
  importSubmitting.value = true
  try {
    await novelStore.createNovelFromText(text, title, {
      genre: '',
      target_genre: '',
      style_tone: '',
      rewrite_strength: '',
      max_chapter_size: maxChapterSize.value,
    })
    importVisible.value = false
    resetImportForm()
    const beforeAutoSplit = novelStore.chapters.length
    const splitReady = await autoSplitOversizedChapters('导入')
    const n = novelStore.chapters.length
    const len = text.length
    if (!splitReady) {
      ElMessage.warning(`已导入 ${len} 字，但仍有内容超过 ${maxChapterSize.value} 字，请重新导入或缩短单段`)
    } else if (n > beforeAutoSplit) {
      ElMessage.success(`已保存 · 已自动拆分为 ${n} 段`)
    } else if (n > 1 && novelStore.splitMode === 'local') {
      ElMessage.success(`已保存 · 自动识别出 ${n} 章`)
    } else if (n === 1) {
      ElMessage.success(`已保存到工作区（${len} 字，单段）`)
      if (['single', 'fallback'].includes(novelStore.splitMode)) {
        ElMessage.warning(
          '未识别到章节标题，已整本导入；如需分章请在原稿中加「第1章」「第2章」等标题后重新导入'
        )
      }
    } else {
      ElMessage.success(`已保存到工作区（${len} 字，${n} 段）`)
    }
  } catch (e) {
    ElMessage.error('导入失败：' + e.message)
  } finally {
    importSubmitting.value = false
  }
}

async function renameCurrentNovel() {
  if (!novelStore.activeNovelId) return
  try {
    const { value } = await ElMessageBox.prompt('小说名称', '重命名', {
      inputValue: novelStore.title,
      inputPlaceholder: '请输入新名称',
      inputValidator: (v) => (v && v.trim() ? true : '名称不能为空'),
      confirmButtonText: '保存',
      cancelButtonText: '取消',
    })
    const next = value.trim()
    if (next && next !== novelStore.title) {
      await novelStore.renameActiveNovel(next)
      ElMessage.success('已重命名')
    }
  } catch {
    /* cancelled */
  }
}

async function switchNovel(id) {
  if (!id || id === novelStore.activeNovelId) return
  try {
    await novelStore.openNovel(id)
    syncRewriteMirrorFromContext(rewriteContexts.value[id] || null)
    restoreRewriteJobs()
  } catch (e) {
    ElMessage.error('打开失败：' + e.message)
  }
}

async function deleteCurrentNovel() {
  if (!novelStore.activeNovelId) return
  try {
    await ElMessageBox.confirm(
      `删除小说「${novelStore.title}」？所有章节和洗稿结果都会被永久删除。`,
      '删除小说',
      { type: 'warning' }
    )
  } catch {
    return
  }
  await novelStore.deleteNovel(novelStore.activeNovelId)
  ElMessage.success('已删除')
}

// ---- rewrite ----
const running = ref(false)
const rawDraft = ref('')
const queueRunning = ref(false)
const queueProgress = ref({ done: 0, total: 0 })
const resultRef = ref(null)
const streamingChars = ref(0) // count of streamed chars for the live header counter
const singleBeforeRun = ref(null)
const runningJob = ref(null)
const activeBatchId = ref(null)
const activeJobIds = ref([])
const activeRewriteNovelId = ref(null)
const activeRewriteNovelTitle = ref('')
const rewritePollTimer = ref(null)
const rewriteJobs = ref([])
const rewriteJobMode = ref('batch')
const rewriteContexts = ref({})
const rewritePollTimers = ref({})

// Unified "is the result panel currently streaming?" — single OR batch.
function contextHasActiveJobs(context) {
  return !!context?.jobs?.some((job) => ['queued', 'running'].includes(job.status))
}

const allRewriteContexts = computed(() => Object.values(rewriteContexts.value))
const currentRewriteContext = computed(() => rewriteContexts.value[novelStore.activeNovelId] || null)
const otherRewriteContexts = computed(() =>
  allRewriteContexts.value.filter((context) =>
    context.novelId !== novelStore.activeNovelId && contextHasActiveJobs(context)
  )
)
const activeRewriteContextCount = computed(() =>
  allRewriteContexts.value.filter((context) => contextHasActiveJobs(context)).length
)
const isStreaming = computed(() =>
  running.value ||
    queueRunning.value ||
    allRewriteContexts.value.some((context) => contextHasActiveJobs(context))
)
const isCurrentNovelStreaming = computed(
  () => contextHasActiveJobs(currentRewriteContext.value) ||
    ((running.value || queueRunning.value) && activeRewriteNovelId.value === novelStore.activeNovelId)
)
const isOtherNovelStreaming = computed(
  () => otherRewriteContexts.value.length > 0 ||
    ((running.value || queueRunning.value) && !!activeRewriteNovelId.value && activeRewriteNovelId.value !== novelStore.activeNovelId)
)
const currentNovelQueueRunning = computed(() => queueRunning.value && isCurrentNovelStreaming.value)
const atParallelNovelLimit = computed(() =>
  !currentRewriteContext.value && activeRewriteContextCount.value >= MAX_PARALLEL_REWRITE_NOVELS
)
const currentRewriteStartDisabled = computed(() =>
  isCurrentNovelStreaming.value || atParallelNovelLimit.value
)

function ensureIdle({ requireSlot = false } = {}) {
  if (isCurrentNovelStreaming.value) {
    ElMessage.warning('当前小说正在生成，请先停止或等待完成')
    return false
  }
  if (requireSlot && atParallelNovelLimit.value) {
    ElMessage.warning(`最多同时洗 ${MAX_PARALLEL_REWRITE_NOVELS} 本小说，请等待其中一本完成后再开始`)
    return false
  }
  return true
}

function oversizedChapters() {
  return novelStore.chapters.filter((c) => (c.content || '').length > maxChapterSize.value)
}

const queueCompleted = computed(() =>
  (queueProgress.value.done || 0) +
    (queueProgress.value.failed || 0) +
    (queueProgress.value.canceled || 0)
)

async function autoSplitOversizedChapters(sourceLabel = '处理') {
  if (!novelStore.activeNovelId) return true
  let oversized = oversizedChapters()
  if (!oversized.length) return true

  const finishedCount = novelStore.chapters.filter(
    (c) => c.status === 'done' || c.script_status === 'done'
  ).length
  if (finishedCount > 0 && sourceLabel !== '导入') {
    try {
      await ElMessageBox.confirm(
        `检测到 ${oversized.length} 段超过 ${maxChapterSize.value} 字，需要先自动拆分。\n\n` +
          `拆分会重置已有的 ${finishedCount} 段处理结果。继续？`,
        '自动拆分',
        { type: 'warning', confirmButtonText: '自动拆分', cancelButtonText: '取消' }
      )
    } catch {
      return false
    }
  }

  try {
    ElMessage.info(
      `${sourceLabel}检测到 ${oversized.length} 段超过 ${maxChapterSize.value} 字，正在自动拆分`
    )
    await novelStore.resplitActiveNovel({
      max_chapter_size: maxChapterSize.value,
    })
    oversized = oversizedChapters()
    if (oversized.length) {
      ElMessage.warning(`${oversized.length} 段仍超过 ${maxChapterSize.value} 字，请缩短后再处理`)
      return false
    }
    return true
  } catch (e) {
    ElMessage.error('自动拆分失败：' + e.message)
    return false
  }
}

const hasUsableModel = computed(() => !!modelStore.activeModel)

async function onSwitchModel(id) {
  if (!id || id === modelStore.activeId) return
  try {
    await modelStore.activate(id)
    ElMessage.success('已切换模型：' + (modelStore.activeModel?.name || ''))
  } catch (e) {
    ElMessage.error('切换模型失败：' + e.message)
  }
}
const novelRewriteMeta = computed(() =>
  [
    novelStore.genre ? `原稿题材：${novelStore.genre}` : '',
    novelStore.target_genre ? `目标题材/世界观：${novelStore.target_genre}` : '',
    novelStore.style_tone ? `文风节奏：${novelStore.style_tone}` : '',
    novelStore.rewrite_strength ? `改写强度：${novelStore.rewrite_strength}` : '',
  ].filter(Boolean).join('\n')
)

// Customer-facing flow stays deliberately small:
//   原稿 (content) → 开始洗稿 → 洗稿 (rewritten) → 导出文稿
const activeChapter = computed(() => novelStore.activeChapter)
const activeChapterContent = computed({
  get: () => activeChapter.value?.content || '',
  set: (val) => activeChapter.value && novelStore.setChapterContent(activeChapter.value.id, val),
})

const rewritePromptOptions = computed(() =>
  promptStore.prompts.filter((p) => p.task !== 'script')
)
const activeRewritePrompt = computed(() => {
  const selected = rewritePromptOptions.value.find((p) => p.id === promptStore.activeId)
  return selected || rewritePromptOptions.value[0] || null
})

const viewVariant = ref('base')

const currentRewritten = computed(() => {
  const c = activeChapter.value
  if (!c) return ''
  if (viewVariant.value === 'source') return c.content || ''
  return c.rewritten || ''
})
const hasBaseRewrite = computed(() => !!activeChapter.value?.rewritten)

const baseRewritePromptId = computed(() => {
  const p = promptStore.prompts.find(
    (x) => x.is_builtin && ['洗稿', '精修'].includes(x.name)
  )
  return activeRewritePrompt.value?.id || p?.id || null
})

function qualityNeedsReview(quality) {
  if (!quality) return false
  if (['excellent', 'pass'].includes(quality.delivery_status)) return false
  return Number(quality.score ?? 100) < 75 || (quality.issues || []).length > 0
}

function overlapDeliveryLabel(v) {
  if (v == null) return ''
  if (v <= OVERLAP_EXCELLENT) return '优秀'
  if (v <= OVERLAP_DELIVERABLE) return '合格'
  return '需复查'
}

function qualityReviewText(quality) {
  if (!qualityNeedsReview(quality)) return ''
  const issues = (quality.issues || []).slice(0, 2).join('；')
  return issues ? `建议复查：${issues}` : '建议复查'
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function generatedExportMissingCount(variant) {
  if (variant === 'source') return 0
  return novelStore.chapters.filter((c) => !c.rewritten).length
}

function chapterQualityIssues(chapter) {
  try {
    return chapter?.quality_issues ? JSON.parse(chapter.quality_issues) : []
  } catch {
    return chapter?.quality_issues ? [chapter.quality_issues] : []
  }
}

function storedChapterNeedsReview(chapter) {
  const issues = chapterQualityIssues(chapter)
  const score = Number(chapter.quality_score ?? 100)
  const grade = chapter.quality_grade || ''
  return score < 75 || issues.length > 0 || (!!grade && !['excellent', 'pass', '优秀', '合格'].includes(grade))
}

function jobQualityIssues(job) {
  return job?.result?.quality?.issues || []
}

function chapterNeedsPolish(chapter) {
  return chapter?.status === 'done' && chapterQualityIssues(chapter).length > 0
}

function jobStatusLabel(job) {
  const status = job?.status || ''
  const phase = job?.phase || ''
  if (phase === 'retry_wait') return '自动重试中'
  if (status === 'queued') return '排队中'
  if (status === 'done') return jobQualityIssues(job).length ? '待完善' : '已完成'
  if (status === 'error') return '失败'
  if (status === 'canceled') return '已取消'
  if (
    phase === 'quality_retry' ||
    phase === 'quality_review' ||
    phase === 'format_retry'
  ) return '质量复查'
  if (phase === 'segment_rewrite' || phase === 'segmenting' || phase === 'merging') return '生成中'
  if (status === 'running') return '生成中'
  return ''
}

function chapterJob(chapter) {
  const jobs = currentRewriteContext.value?.jobs || rewriteJobs.value
  return jobs.find((job) => job.chapter_id === chapter?.id) || null
}

function chapterStatusLabel(chapter) {
  const job = chapterJob(chapter)
  if (job) return jobStatusLabel(job)
  if (chapterNeedsPolish(chapter)) return '待完善'
  if (chapter?.status === 'done') return '已完成'
  if (chapter?.status === 'error') return '失败'
  if (chapter?.status === 'queued') return '排队中'
  if (chapter?.status === 'running') return '生成中'
  if (chapter?.status === 'canceled') return '已取消'
  return ''
}

function chapterForRewriteJob(job) {
  return novelStore.chapters.find((chapter) => chapter.id === job?.chapter_id) || null
}

function estimateRewriteSeconds(chapter, job) {
  const payloadText = job?.payload?.text || ''
  const sourceLen = (chapter?.content || payloadText || '').length
  const len = Math.max(120, sourceLen)
  const segmentCount = len > DEFAULT_MAX_CHAPTER_SIZE ? Math.max(2, Math.ceil(len / DEFAULT_MAX_CHAPTER_SIZE)) : 1
  const baseSeconds = Math.ceil(len / 420) * 55 + segmentCount * 35
  return Math.min(18 * 60, Math.max(90, baseSeconds))
}

function formatEta(seconds) {
  const value = Math.max(0, Math.ceil(Number(seconds) || 0))
  if (!value) return ''
  if (value < 60) return '预计 1 分钟内'
  const minutes = Math.ceil(value / 60)
  if (minutes < 60) return `预计约 ${minutes} 分钟`
  const hours = Math.floor(minutes / 60)
  const rest = minutes % 60
  return rest ? `预计约 ${hours} 小时 ${rest} 分钟` : `预计约 ${hours} 小时`
}

function jobRemainingSeconds(job) {
  if (!job || ['done', 'error', 'canceled'].includes(job.status)) return 0
  const estimate = estimateRewriteSeconds(chapterForRewriteJob(job), job)
  const phase = job.phase || ''
  if (['quality_retry', 'format_retry', 'quality_review'].includes(phase)) {
    return Math.max(120, Math.ceil(estimate * 0.65))
  }
  if (phase === 'retry_wait') return Math.max(60, Math.ceil(estimate * 0.75))
  const progress = Math.max(0, Math.min(96, Number(job.progress || 0)))
  if (job.status === 'queued') return estimate
  return Math.max(30, Math.ceil(estimate * (1 - progress / 100)))
}

const activeRewriteJob = computed(() => {
  const selected = chapterJob(activeChapter.value)
  if (selected && ['queued', 'running'].includes(selected.status)) return selected
  const context = currentRewriteContext.value
  const activeSet = new Set(context?.jobIds || activeJobIds.value)
  const jobs = context?.jobs || rewriteJobs.value
  const candidates = jobs.filter((job) =>
    ['queued', 'running'].includes(job.status) &&
      (!activeSet.size || activeSet.has(job.id))
  )
  return candidates.find((job) => job.status === 'running') || candidates[0] || null
})

const queueRemainingSeconds = computed(() => {
  const jobs = currentRewriteContext.value?.jobs || rewriteJobs.value
  const active = jobs.filter((job) => ['queued', 'running'].includes(job.status))
  if (active.length) {
    return active.reduce((sum, job) => sum + jobRemainingSeconds(job), 0)
  }
  const pendingChapters = novelStore.chapters.filter((chapter) =>
    ['queued', 'running'].includes(chapter.status)
  )
  return pendingChapters.reduce((sum, chapter) => sum + estimateRewriteSeconds(chapter), 0)
})

const queueEtaText = computed(() => {
  if (!queueRunning.value) return ''
  return formatEta(queueRemainingSeconds.value)
})

const queueStageText = computed(() => {
  const jobs = currentRewriteContext.value?.jobs || rewriteJobs.value
  const current = jobs.find((job) => job.status === 'running') ||
    jobs.find((job) => job.status === 'queued')
  if (current) return jobStatusLabel(current) || '生成中'
  return '准备生成'
})

const rewriteActivityText = computed(() => {
  const job = activeRewriteJob.value
  const label = job ? (jobStatusLabel(job) || '生成中') : (queueRunning.value ? queueStageText.value : '生成中')
  const eta = job ? formatEta(jobRemainingSeconds(job)) : queueEtaText.value
  return eta ? `${label} · ${eta}` : label
})

function contextProgress(context) {
  const jobs = context?.jobs || []
  const done = jobs.filter((job) => job.status === 'done').length
  const failed = jobs.filter((job) => job.status === 'error').length
  const canceled = jobs.filter((job) => job.status === 'canceled').length
  const review = jobs.filter((job) => {
    const issues = job.result?.quality?.issues || []
    return job.status === 'done' && issues.length > 0
  }).length
  return { done, failed, canceled, review, total: jobs.length }
}

function contextQueueCompleted(context) {
  const progress = contextProgress(context)
  return (progress.done || 0) + (progress.failed || 0) + (progress.canceled || 0)
}

function contextActiveJob(context) {
  const jobs = context?.jobs || []
  const activeSet = new Set(context?.jobIds || [])
  const candidates = jobs.filter((job) =>
    ['queued', 'running'].includes(job.status) &&
      (!activeSet.size || activeSet.has(job.id))
  )
  return candidates.find((job) => job.status === 'running') || candidates[0] || null
}

function contextRemainingSeconds(context) {
  const active = (context?.jobs || []).filter((job) => ['queued', 'running'].includes(job.status))
  return active.reduce((sum, job) => sum + jobRemainingSeconds(job), 0)
}

function contextActivityText(context) {
  const job = contextActiveJob(context)
  const label = job ? (jobStatusLabel(job) || '生成中') : '生成中'
  const eta = job ? formatEta(jobRemainingSeconds(job)) : formatEta(contextRemainingSeconds(context))
  return eta ? `${label} · ${eta}` : label
}

function backgroundRewriteText(context) {
  const title = context?.novelTitle || activeRewriteNovelTitle.value || '其他小说'
  const progressValue = contextProgress(context)
  const progress = context?.mode !== 'single' && progressValue.total
    ? ` · ${contextQueueCompleted(context)}/${progressValue.total}`
    : ''
  return `${title}${progress} · ${contextActivityText(context)}`
}

async function ensureRewriteAnalysisReady() {
  const novelId = novelStore.activeNovelId
  if (!novelId) return false
  if (novelStore.analysisStatus === 'done') return true

  ElMessage.info('正在整理人物/世界观/情节线，完成后自动开始洗稿')
  try {
    await api.reanalyzeNovel(novelId)
    if (novelStore.activeNovelId === novelId) novelStore.analysisStatus = 'running'
  } catch (e) {
    ElMessage.error('人物/世界观/情节线整理启动失败：' + e.message)
    return false
  }

  const deadline = Date.now() + REWRITE_ANALYSIS_WAIT_MS
  while (Date.now() < deadline) {
    await sleep(3000)
    let status = 'idle'
    try {
      const novels = await novelStore.refreshNovelList()
      const current = novels.find((novel) => novel.id === novelId)
      status = current?.analysis_status || 'idle'
    } catch (e) {
      console.warn('rewrite analysis wait failed', e)
      continue
    }
    if (novelStore.activeNovelId === novelId) novelStore.analysisStatus = status
    if (status === 'done') return true
    if (status === 'error') {
      ElMessage.error('人物/世界观/情节线整理失败，请重新整理后再试')
      return false
    }
  }
  ElMessage.warning('人物/世界观/情节线仍在整理，请稍后再开始')
  return false
}

function readPersistedRewriteContext() {
  try {
    const raw = window.localStorage.getItem(ACTIVE_REWRITE_CONTEXT_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : [parsed]
  } catch {
    return []
  }
}

function clearPersistedRewriteContext() {
  try {
    window.localStorage.removeItem(ACTIVE_REWRITE_CONTEXT_KEY)
  } catch {
    // localStorage can be disabled in private / embedded browser contexts.
  }
}

function persistActiveRewriteContext() {
  const contexts = Object.values(rewriteContexts.value)
    .filter((context) => contextHasActiveJobs(context))
    .map((context) => ({
      novel_id: context.novelId,
      novel_title: context.novelTitle,
      batch_id: context.batchId,
      job_ids: context.jobIds,
      mode: context.mode,
    }))
  if (!contexts.length) {
    clearPersistedRewriteContext()
    return
  }
  try {
    window.localStorage.setItem(
      ACTIVE_REWRITE_CONTEXT_KEY,
      JSON.stringify(contexts)
    )
  } catch {
    // Persistence is a recovery aid; polling still works in the current tab.
  }
}

function novelTitleForRewriteContext(novelId, fallback = '') {
  return fallback ||
    novelStore.novels.find((novel) => novel.id === novelId)?.title ||
    (novelId === novelStore.activeNovelId ? novelStore.title : '') ||
    '其他小说'
}

function syncRewriteMirrorFromContext(context) {
  if (!context) {
    running.value = false
    queueRunning.value = false
    queueProgress.value = { done: 0, total: 0, failed: 0, review: 0, canceled: 0 }
    activeJobIds.value = []
    activeBatchId.value = null
    activeRewriteNovelId.value = null
    activeRewriteNovelTitle.value = ''
    rewriteJobs.value = []
    rewriteJobMode.value = 'batch'
    return
  }
  activeBatchId.value = context.batchId || null
  activeJobIds.value = context.jobIds || []
  activeRewriteNovelId.value = context.novelId
  activeRewriteNovelTitle.value = context.novelTitle || ''
  rewriteJobMode.value = context.mode || 'batch'
  rewriteJobs.value = context.jobs || []
  queueProgress.value = contextProgress(context)
  running.value = context.mode === 'single' && contextHasActiveJobs(context)
  queueRunning.value = context.mode !== 'single' && contextHasActiveJobs(context)
  runningJob.value = context.runningJob || runningJob.value
  singleBeforeRun.value = context.singleBeforeRun || singleBeforeRun.value
}

function upsertRewriteContext(patch) {
  const novelId = patch?.novelId
  if (!novelId) return null
  const previous = rewriteContexts.value[novelId] || {}
  const next = {
    ...previous,
    ...patch,
    novelId,
    novelTitle: novelTitleForRewriteContext(novelId, patch.novelTitle || previous.novelTitle),
    jobIds: patch.jobIds || previous.jobIds || [],
    jobs: patch.jobs || previous.jobs || [],
    mode: patch.mode || previous.mode || 'batch',
  }
  rewriteContexts.value = { ...rewriteContexts.value, [novelId]: next }
  if (novelId === novelStore.activeNovelId) syncRewriteMirrorFromContext(next)
  persistActiveRewriteContext()
  return next
}

function removeRewriteContext(novelId, { clearPersisted = false } = {}) {
  if (!novelId) return
  clearRewritePollTimer(novelId)
  const next = { ...rewriteContexts.value }
  delete next[novelId]
  rewriteContexts.value = next
  if (novelId === novelStore.activeNovelId) syncRewriteMirrorFromContext(null)
  if (clearPersisted) clearPersistedRewriteContext()
  else persistActiveRewriteContext()
}

function resetActiveRewriteContext({ clearJobs = true, clearPersisted = true } = {}) {
  const novelId = activeRewriteNovelId.value || novelStore.activeNovelId
  if (novelId) removeRewriteContext(novelId, { clearPersisted })
  activeJobIds.value = []
  activeBatchId.value = null
  activeRewriteNovelId.value = null
  activeRewriteNovelTitle.value = ''
  if (clearJobs) rewriteJobs.value = []
  if (clearPersisted) clearPersistedRewriteContext()
}

function applyRewriteJobSnapshot(jobs, targetNovelId = activeRewriteNovelId.value || novelStore.activeNovelId) {
  const snapshotJobs = jobs || []
  if (targetNovelId && rewriteContexts.value[targetNovelId]) {
    upsertRewriteContext({ novelId: targetNovelId, jobs: snapshotJobs })
  }
  if (targetNovelId !== novelStore.activeNovelId) return
  rewriteJobs.value = snapshotJobs
  queueProgress.value = contextProgress({ jobs: snapshotJobs })
  for (const job of snapshotJobs) {
    const chapter = novelStore.chapters.find((c) => c.id === job.chapter_id)
    if (!chapter) continue
    chapter.rewrite_phase = job.phase || ''
    chapter.rewrite_status_label = jobStatusLabel(job)
    if (job.status === 'queued') {
      chapter.status = 'queued'
    } else if (job.status === 'running') {
      chapter.status = 'running'
    } else if (job.status === 'error') {
      chapter.status = 'error'
    } else if (job.status === 'canceled') {
      chapter.status = 'canceled'
    } else if (job.status === 'done') {
      chapter.status = 'done'
      const result = job.result || {}
      const text = result.rewritten || ''
      if (text) {
        chapter.rewritten = text
        if (result.quality) {
          chapter.overlap = result.quality.overlap4 ?? chapter.overlap
          chapter.quality_score = Number(result.quality.score ?? chapter.quality_score ?? 0)
          chapter.quality_grade =
            result.quality.delivery_status || result.quality.delivery_label || result.quality.grade || ''
          chapter.quality_issues = JSON.stringify(result.quality.issues || [])
        }
      }
    }
  }
}

function clearRewritePollTimer(novelId = null) {
  if (novelId) {
    const timer = rewritePollTimers.value[novelId]
    if (timer) clearTimeout(timer)
    const next = { ...rewritePollTimers.value }
    delete next[novelId]
    rewritePollTimers.value = next
    return
  }
  if (rewritePollTimer.value) clearTimeout(rewritePollTimer.value)
  rewritePollTimer.value = null
  for (const timer of Object.values(rewritePollTimers.value)) {
    clearTimeout(timer)
  }
  rewritePollTimers.value = {}
}

function activeJobsFinished(jobs) {
  return jobs.length > 0 && jobs.every((job) => ['done', 'error', 'canceled'].includes(job.status))
}

function scheduleRewriteContextPoll(novelId) {
  clearRewritePollTimer(novelId)
  const timer = setTimeout(() => pollRewriteContext(novelId), REWRITE_JOB_POLL_MS)
  rewritePollTimers.value = { ...rewritePollTimers.value, [novelId]: timer }
}

async function pollRewriteContext(novelId) {
  const context = rewriteContexts.value[novelId]
  if (!context || (!context.batchId && !context.jobIds?.length)) return
  try {
    const res = await api.listNovelRewriteJobs(
      novelId,
      context.batchId ? { batch_id: context.batchId } : {}
    )
    let jobs = res.jobs || []
    if (!context.batchId && context.jobIds?.length) {
      const activeSet = new Set(context.jobIds)
      jobs = jobs.filter((job) => activeSet.has(job.id))
    }
    upsertRewriteContext({ novelId, jobs })
    applyRewriteJobSnapshot(jobs, novelId)
    if (activeJobsFinished(jobs)) {
      const failed = jobs.filter((job) => job.status === 'error').length
      const canceled = jobs.filter((job) => job.status === 'canceled').length
      const review = contextProgress({ jobs }).review || 0
      const finishedContext = rewriteContexts.value[novelId] || context
      removeRewriteContext(novelId)
      if (novelId === novelStore.activeNovelId) {
        await novelStore.openNovel(novelId)
      }
      await novelStore.refreshNovelList()
      const prefix = finishedContext.novelTitle ? `「${finishedContext.novelTitle}」` : ''
      if (finishedContext.mode === 'single') {
        if (failed) ElMessage.error(`${prefix}洗稿失败，请稍后重试`)
        else if (canceled) ElMessage.warning(`${prefix}已取消洗稿`)
        else if (review) ElMessage.warning(`${prefix}洗稿已完成，建议人工快速浏览后再导出`)
        else ElMessage.success(`${prefix}洗稿完成`)
      } else if (failed) {
        ElMessage.warning(`${prefix}整本洗稿完成，${failed} 章失败。可点击重试失败章节`)
      } else if (canceled) {
        ElMessage.warning(`${prefix}已取消 ${canceled} 个任务`)
      } else if (review) {
        ElMessage.warning(`${prefix}整本洗稿完成，${review} 章建议人工快速浏览`)
      } else {
        ElMessage.success(`${prefix}整本洗稿完成`)
      }
      return
    }
  } catch (e) {
    console.warn('rewrite job poll failed', e)
  }
  if (contextHasActiveJobs(rewriteContexts.value[novelId])) {
    scheduleRewriteContextPoll(novelId)
  }
}

async function pollRewriteJobs() {
  const pollNovelId = activeRewriteNovelId.value || novelStore.activeNovelId
  if (!pollNovelId) return
  if (!rewriteContexts.value[pollNovelId] && (activeBatchId.value || activeJobIds.value.length)) {
    upsertRewriteContext({
      novelId: pollNovelId,
      novelTitle: activeRewriteNovelTitle.value || novelStore.title,
      batchId: activeBatchId.value,
      jobIds: activeJobIds.value,
      jobs: rewriteJobs.value,
      mode: rewriteJobMode.value,
    })
  }
  await pollRewriteContext(pollNovelId)
}

async function startRewriteJobPolling(
  batchId,
  jobs,
  mode = 'batch',
  novelId = novelStore.activeNovelId,
  novelTitle = novelStore.title
) {
  const jobIds = (jobs || []).map((job) => job.id).filter(Boolean)
  const activeJob = mode === 'single'
    ? ((jobs || []).find((job) => ['queued', 'running'].includes(job.status)) || (jobs || [])[0])
    : null
  const context = upsertRewriteContext({
    novelId,
    novelTitle,
    batchId,
    jobIds,
    jobs: jobs || [],
    mode,
    runningJob: activeJob?.chapter_id ? { chapterId: activeJob.chapter_id, variant: 'base' } : null,
    singleBeforeRun: mode === 'single' ? singleBeforeRun.value : null,
  })
  if (mode === 'single') {
    runningJob.value = activeJob?.chapter_id ? { chapterId: activeJob.chapter_id, variant: 'base' } : runningJob.value
  }
  if (context && novelId === novelStore.activeNovelId) syncRewriteMirrorFromContext(context)
  applyRewriteJobSnapshot(jobs || [], novelId)
  await pollRewriteContext(novelId)
}

async function restoreRewriteJobs() {
  const persisted = readPersistedRewriteContext()
  const candidates = []
  for (const context of persisted) candidates.push(context)
  if (novelStore.activeNovelId && !candidates.some((candidate) => candidate.novel_id === novelStore.activeNovelId)) {
    candidates.push({ novel_id: novelStore.activeNovelId })
  }
  if (!candidates.length) return
  try {
    for (const candidate of candidates) {
      const novelId = candidate.novel_id
      const requestOptions = candidate.batch_id ? { batch_id: candidate.batch_id } : { active: true }
      const res = await api.listNovelRewriteJobs(novelId, requestOptions)
      let jobs = res.jobs || []
      if (candidate.job_ids?.length) {
        const ids = new Set(candidate.job_ids)
        jobs = jobs.filter((job) => ids.has(job.id))
      }
      const activeJobs = jobs.filter((job) => ['queued', 'running'].includes(job.status))
      if (!activeJobs.length) {
        continue
      }
      const batchIds = Array.from(new Set(activeJobs.map((job) => job.batch_id).filter(Boolean)))
      const batchId = candidate.batch_id || (batchIds.length === 1 ? batchIds[0] : null)
      const mode = candidate.mode || (activeJobs.length === 1 && jobs.length === 1 ? 'single' : 'batch')
      await startRewriteJobPolling(
        batchId,
        jobs,
        mode,
        novelId,
        novelTitleForRewriteContext(novelId, candidate.novel_title)
      )
    }
  } catch (e) {
    console.warn('restore rewrite jobs failed', e)
  }
}

function rewriteJobPayload(chapter) {
  return {
    prompt_id: baseRewritePromptId.value,
    model_id: modelStore.activeId,
    plot_hint: chapter?.summary || '',
    genre_hint: novelRewriteMeta.value,
    quality_mode: DEFAULT_REWRITE_QUALITY_MODE,
  }
}

async function generatedExportReviewCount(variant) {
  if (variant === 'source') return 0
  let count = 0
  for (const chapter of novelStore.chapters) {
    if (!chapter.rewritten) continue
    try {
      const quality = await api.qualityScore({
        rewritten: chapter.rewritten,
        source: chapter.content || '',
      })
      if (qualityNeedsReview(quality)) count += 1
    } catch {
      if (storedChapterNeedsReview(chapter)) count += 1
    }
  }
  return count
}

watch(activeChapter, (c) => {
  rawDraft.value = c?.rewritten || ''
  streamingChars.value = c?.rewritten?.length || 0
})

// Auto-scroll the result panel as new chunks stream in (single OR batch).
watch(
  () => currentRewritten.value,
  () => {
    if (!isCurrentNovelStreaming.value) return
    nextTick(() => {
      const el = resultRef.value
      if (!el) return
      // Only autoscroll if the user is already near the bottom — don't yank
      // their view if they've scrolled up to read.
      const nearBottom = el.scrollHeight - el.clientHeight - el.scrollTop < 120
      if (nearBottom) el.scrollTop = el.scrollHeight
    })
  }
)

async function startRewrite() {
  if (!ensureIdle({ requireSlot: true })) return
  if (!activeChapter.value) {
    ElMessage.warning('请先选择章节')
    return
  }
  if (!activeChapter.value.content.trim()) {
    ElMessage.warning('当前章节没有内容')
    return
  }
  if (!hasUsableModel.value) {
    ElMessage.warning('请先激活一个模型')
    return
  }
  if (!baseRewritePromptId.value) {
    ElMessage.warning(rewriteCapabilityMissingMessage)
    return
  }
  if (!activeRewritePrompt.value) {
    ElMessage.warning(rewriteCapabilityMissingMessage)
    return
  }
  if (!(await ensureRewriteAnalysisReady())) return

  const chapter = activeChapter.value
  if (!chapter?.content?.trim()) {
    ElMessage.warning('当前章节没有内容')
    return
  }
  const startNovelId = novelStore.activeNovelId
  const startNovelTitle = novelStore.title
  singleBeforeRun.value = {
    id: chapter.id,
    rewritten: chapter.rewritten || '',
    status: chapter.status || 'idle',
    overlap: chapter.overlap,
  }
  runningJob.value = { chapterId: chapter.id, variant: 'base' }
  activeRewriteNovelId.value = startNovelId
  activeRewriteNovelTitle.value = startNovelTitle || ''
  const promptId = baseRewritePromptId.value
  viewVariant.value = 'base'
  running.value = true
  rawDraft.value = ''
  streamingChars.value = 0
  novelStore.setRewritten(chapter.id, chapter.rewritten || '', chapter.overlap, 'queued', false, 'base', null)

  try {
    const job = await api.enqueueChapterRewrite(chapter.id, {
      ...rewriteJobPayload(chapter),
      prompt_id: promptId,
    })
    await startRewriteJobPolling(job.batch_id, [job], 'single', startNovelId, startNovelTitle)
  } catch (e) {
    running.value = false
    resetActiveRewriteContext()
    const prev = singleBeforeRun.value
    novelStore.setRewritten(
      chapter.id,
      prev?.rewritten || '',
      prev?.overlap,
      'error',
      false,
      'base'
    )
    singleBeforeRun.value = null
    runningJob.value = null
    ElMessage.error('洗稿入队失败：' + e.message)
  }
}

async function stopRewrite() {
  const rewriteNovelId = activeRewriteNovelId.value
  await cancelActiveRewriteJobs()
  running.value = false
  if (rewriteNovelId === novelStore.activeNovelId) {
    const job = runningJob.value
    const chapter = job?.chapterId
      ? novelStore.chapters.find((c) => c.id === job.chapterId)
      : null
    if (chapter && singleBeforeRun.value?.id === chapter.id) {
      const prev = singleBeforeRun.value
      novelStore.setRewritten(
        chapter.id,
        prev?.rewritten || chapter.rewritten,
        prev?.overlap,
        prev?.status || 'idle'
      )
    } else if (rewriteNovelId) {
      await novelStore.openNovel(rewriteNovelId)
    }
  }
  singleBeforeRun.value = null
  runningJob.value = null
  if (rewriteNovelId) removeRewriteContext(rewriteNovelId)
}

async function stopActiveRewrite(context = null) {
  if (context?.novelId && context.novelId !== novelStore.activeNovelId) {
    await cancelActiveRewriteJobs(context.novelId)
    removeRewriteContext(context.novelId)
  } else if (queueRunning.value) {
    await stopQueue()
  } else if (running.value) {
    await stopRewrite()
  }
}

async function rewriteAll() {
  if (!ensureIdle({ requireSlot: true })) return
  if (!novelStore.chapters.length) return
  if (!hasUsableModel.value) {
    ElMessage.warning('请先激活一个模型')
    return
  }
  if (!baseRewritePromptId.value) {
    ElMessage.warning(rewriteCapabilityMissingMessage)
    return
  }
  if (!(await ensureRewriteAnalysisReady())) return
  const doneCount = novelStore.chapters.filter((c) => c.status === 'done').length
  const total = novelStore.chapters.length
  const unfinishedCount = total - doneCount

  // Three-way choice when some chapters already done: skip / overwrite / cancel.
  // When nothing done yet, just a single info confirm.
  let onlyUnfinished = false
  if (doneCount > 0 && unfinishedCount > 0) {
    let choice
    try {
      choice = await ElMessageBox.confirm(
        `共 ${total} 章，已洗 ${doneCount} 章，未洗 ${unfinishedCount} 章。\n\n` +
          `选择处理方式：\n` +
          `· 仅跑未洗的 ${unfinishedCount} 章（推荐，省 token）\n` +
          `· 全部覆盖重洗（已洗的会被新结果替换，无法恢复）`,
        '整本洗稿',
        {
          type: 'info',
          confirmButtonText: `仅跑未洗 ${unfinishedCount} 章`,
          cancelButtonText: '全部覆盖重洗',
          distinguishCancelAndClose: true,
        }
      )
      onlyUnfinished = true
    } catch (action) {
      // distinguishCancelAndClose: 'cancel' = clicked 全部覆盖, 'close' = closed dialog
      if (action === 'cancel') {
        try {
          await ElMessageBox.confirm(
            `⚠️ 已洗的 ${doneCount} 章洗稿结果会被覆盖，无法恢复。继续？`,
            '确认覆盖',
            { type: 'warning', confirmButtonText: '确认覆盖', cancelButtonText: '返回' }
          )
          onlyUnfinished = false
        } catch {
          return
        }
      } else {
        return
      }
    }
  } else if (doneCount === total) {
    try {
      await ElMessageBox.confirm(
        `全部 ${total} 章都已洗过。重洗会覆盖现有结果。继续？`,
        '整本洗稿',
        { type: 'warning', confirmButtonText: '全部重洗', cancelButtonText: '取消' }
      )
    } catch { return }
  } else {
    // 0 done
    try {
      await ElMessageBox.confirm(
        `即将对全部 ${total} 章逐章洗稿。\n\n中途可点右上「停止批量」中断。继续？`,
        '整本洗稿',
        { type: 'info', confirmButtonText: '开始整本洗稿', cancelButtonText: '取消' }
      )
    } catch { return }
  }

  queueRunning.value = true
  activeRewriteNovelId.value = novelStore.activeNovelId
  activeRewriteNovelTitle.value = novelStore.title || ''
  queueProgress.value = { done: 0, total: 0, failed: 0, review: 0, canceled: 0 }
  const startNovelId = novelStore.activeNovelId
  const startNovelTitle = novelStore.title
  const variant = 'base'
  const promptId = baseRewritePromptId.value
  viewVariant.value = variant
  try {
    const res = await api.enqueueNovelRewrite(novelStore.activeNovelId, {
      prompt_id: promptId,
      model_id: modelStore.activeId,
      genre_hint: novelRewriteMeta.value,
      quality_mode: DEFAULT_REWRITE_QUALITY_MODE,
      only_unfinished: onlyUnfinished,
      overwrite: !onlyUnfinished,
    })
    if (!res.jobs?.length) {
      queueRunning.value = false
      resetActiveRewriteContext()
      ElMessage.info('没有需要洗稿的章节')
      return
    }
    await startRewriteJobPolling(res.batch_id, res.jobs, 'batch', startNovelId, startNovelTitle)
  } catch (e) {
    queueRunning.value = false
    resetActiveRewriteContext()
    ElMessage.error('整本洗稿入队失败：' + e.message)
  }
}

/** Re-run rewrite on only the chapters that previously errored.
 *  Useful after an integral 整本洗稿 partially failed. */
async function retryFailedChapters() {
  if (!ensureIdle({ requireSlot: true })) return
  const failed = novelStore.chapters.filter((c) => c.status === 'error')
  if (!failed.length) return
  if (!hasUsableModel.value) {
    ElMessage.warning('请先激活一个模型')
    return
  }
  if (!baseRewritePromptId.value) {
    ElMessage.warning(rewriteCapabilityMissingMessage)
    return
  }
  if (!(await ensureRewriteAnalysisReady())) return
  queueRunning.value = true
  activeRewriteNovelId.value = novelStore.activeNovelId
  activeRewriteNovelTitle.value = novelStore.title || ''
  queueProgress.value = { done: 0, total: failed.length, failed: 0, review: 0, canceled: 0 }
  const startNovelId = novelStore.activeNovelId
  const startNovelTitle = novelStore.title
  const variant = 'base'
  const promptId = baseRewritePromptId.value
  viewVariant.value = variant
  try {
    const res = await api.enqueueNovelRewrite(novelStore.activeNovelId, {
      prompt_id: promptId,
      model_id: modelStore.activeId,
      genre_hint: novelRewriteMeta.value,
      quality_mode: DEFAULT_REWRITE_QUALITY_MODE,
      only_failed: true,
    })
    if (!res.jobs?.length) {
      queueRunning.value = false
      resetActiveRewriteContext()
      ElMessage.info('没有失败章节需要重试')
      return
    }
    await startRewriteJobPolling(res.batch_id, res.jobs, 'batch', startNovelId, startNovelTitle)
  } catch (e) {
    queueRunning.value = false
    resetActiveRewriteContext()
    ElMessage.error('重试入队失败：' + e.message)
  }
}

const failedCount = computed(
  () => novelStore.chapters.filter((c) => c.status === 'error').length
)

// ---- Inline chapter rename ----
const editingChapterId = ref(null)
const editingChapterTitle = ref('')
function startEditChapterTitle(c) {
  editingChapterId.value = c.id
  editingChapterTitle.value = c.title
  nextTick(() => {
    const el = document.querySelector('.chapter-title-input')
    if (el) {
      el.focus()
      el.select()
    }
  })
}
async function commitChapterTitle(c) {
  if (editingChapterId.value !== c.id) return
  const next = editingChapterTitle.value
  editingChapterId.value = null
  if (next && next.trim() && next.trim() !== c.title) {
    await novelStore.renameChapter(c.id, next)
  }
}
function cancelChapterTitle() {
  editingChapterId.value = null
  editingChapterTitle.value = ''
}

async function cancelActiveRewriteJobs(novelId = activeRewriteNovelId.value || novelStore.activeNovelId) {
  const context = rewriteContexts.value[novelId]
  const ids = context?.jobIds?.length ? [...context.jobIds] : [...activeJobIds.value]
  if (novelId === activeRewriteNovelId.value) activeJobIds.value = []
  clearRewritePollTimer(novelId)
  await Promise.allSettled(ids.map(async (id) => {
    await api.cancelRewriteJob(id)
  }))
}

async function stopQueue() {
  const rewriteNovelId = activeRewriteNovelId.value
  queueRunning.value = false
  await cancelActiveRewriteJobs()
  if (rewriteNovelId) removeRewriteContext(rewriteNovelId)
  if (rewriteNovelId === novelStore.activeNovelId) {
    await novelStore.openNovel(novelStore.activeNovelId)
  }
}

const origLen = computed(() => (activeChapter.value?.content || '').length)

const justCopied = ref(false)
async function copyRewritten() {
  const text = currentRewritten.value
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
  } catch (e) {
    ElMessage.error('复制失败：' + e.message)
    return
  }
  justCopied.value = true
  ElMessage.success('已复制到剪贴板')
  setTimeout(() => { justCopied.value = false }, 1800)
}

const paneTitle = computed(() => {
  if (viewVariant.value === 'source') return '原稿'
  return '洗稿结果'
})

const canExportSource = computed(() => novelStore.chapters.some((c) => c.content))
const canExportBase = computed(() => novelStore.chapters.some((c) => c.rewritten))
const exportDisabled = computed(() => {
  if (!novelStore.chapters.length) return true
  return !canExportSource.value && !canExportBase.value
})

async function handleExportCommand(command) {
  await exportAll(command)
}

async function exportAll(selectedVariant = viewVariant.value) {
  if (!novelStore.chapters.length) return
  const variant = selectedVariant
  let field, suffix
  if (variant === 'source') {
    field = 'content'; suffix = '原稿'
  } else {
    field = 'rewritten'; suffix = '洗稿'
  }
  const hasContent =
    variant === 'source'
      ? canExportSource.value
      : canExportBase.value
  if (!hasContent) {
    ElMessage.warning(`${suffix}还没有可导出的内容`)
    return
  }
  const missingCount = generatedExportMissingCount(variant)
  if (missingCount > 0) {
    ElMessage.warning(`洗稿还有 ${missingCount} 章未生成，先生成完整后再导出`)
    return
  }
  const reviewCount = await generatedExportReviewCount(variant)
  if (reviewCount > 0) {
    try {
      await ElMessageBox.confirm(
        `仍有 ${reviewCount} 章待完善，建议人工快速浏览后再交付。是否继续导出？`,
        '导出提醒',
        {
          type: 'warning',
          confirmButtonText: '继续导出',
          cancelButtonText: '返回检查',
        }
      )
    } catch {
      return
    }
  }
  const out = novelStore.chapters
    .map((c) => {
      const value = c[field]
      return `${c.title}\n\n${value || ''}`
    })
    .join('\n\n\n')
  const blob = new Blob([out], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${novelStore.title || 'rewritten'}_${suffix}.txt`
  a.click()
  URL.revokeObjectURL(url)
  ElMessage.success(`已导出${suffix}全集`)
}

// ---- Keyboard shortcuts ----
// Cmd/Ctrl+Enter — start (or stop) rewrite
// Esc — stop running rewrite
function handleKeydown(e) {
  // Don't hijack typing inside form fields except for the Esc-to-stop case.
  const tag = (e.target?.tagName || '').toLowerCase()
  const inField = tag === 'input' || tag === 'textarea' || e.target?.isContentEditable

  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault()
    if (isCurrentNovelStreaming.value && running.value) {
      stopRewrite()
    } else if (isCurrentNovelStreaming.value && queueRunning.value) {
      stopQueue()
    } else if (activeChapter.value && hasUsableModel.value) {
      startRewrite()
    }
    return
  }
  if (e.key === 'Escape' && !inField) {
    if (isOtherNovelStreaming.value) return
    if (isCurrentNovelStreaming.value && running.value) stopRewrite()
    else if (isCurrentNovelStreaming.value && queueRunning.value) stopQueue()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
  restoreRewriteJobs()
})

onUnmounted(() => {
  clearRewritePollTimer()
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div class="workbench">
    <!-- Left sidebar -->
    <aside class="sidebar">
      <div class="sidebar-head">
        <div class="section-title">我的小说</div>
        <el-button size="small" type="primary" @click="openImport">
          <el-icon><Plus /></el-icon>
          <span style="margin-left: 4px">导入小说</span>
        </el-button>
      </div>

      <div v-if="novelStore.novels.length" class="novel-picker">
        <el-select
          :model-value="novelStore.activeNovelId"
          @update:model-value="switchNovel"
          size="default"
          placeholder="选择小说"
          class="novel-select"
        >
          <el-option
            v-for="n in novelStore.novels"
            :key="n.id"
            :value="n.id"
            :label="n.title"
          >
            <div class="novel-opt">
              <span class="novel-opt-title">{{ n.title }}</span>
              <span class="novel-opt-meta">
                {{ n.chapter_count }}章 · {{ n.done_count || 0 }} 已洗
              </span>
            </div>
          </el-option>
        </el-select>
        <el-tooltip content="重命名当前小说">
          <el-button
            size="small"
            :icon="EditPen"
            :disabled="!novelStore.activeNovelId"
            @click="renameCurrentNovel"
          />
        </el-tooltip>
        <el-tooltip content="删除当前小说">
          <el-button
            size="small"
            type="danger"
            plain
            :icon="Delete"
            :disabled="!novelStore.activeNovelId"
            @click="deleteCurrentNovel"
          />
        </el-tooltip>
      </div>

      <!-- Auto-save + analysis indicator -->
      <div v-if="novelStore.activeNovelId" class="save-status">
        <transition name="fade" mode="out-in">
          <span v-if="novelStore.saving" key="save" class="save-status-saving">
            <el-icon class="rotate"><Loading /></el-icon>
            保存中…
          </span>
          <span v-else-if="novelStore.analysisStatus === 'running'" key="ana" class="save-status-saving">
            <el-icon class="rotate"><Loading /></el-icon>
            正在整理文本…
          </span>
          <span v-else-if="novelStore.analysisStatus === 'error'" key="err" class="save-status-err">
            <el-icon><Check /></el-icon>
            已自动保存
          </span>
          <span v-else-if="novelStore.analysisStatus === 'done'" key="ok-ana" class="save-status-ok">
            <el-icon><Check /></el-icon>
            已自动保存
          </span>
          <span v-else key="ok" class="save-status-ok">
            <el-icon><Check /></el-icon>
            已自动保存
          </span>
        </transition>
      </div>

      <div v-if="!novelStore.chapters.length" class="empty-state">
        <div class="empty-state-icon">
          <el-icon :size="32"><Document /></el-icon>
        </div>
        <div class="empty-state-title">还没有原稿</div>
        <div class="empty-state-desc">点击「导入小说」粘贴或上传 TXT / DOCX 文件</div>
      </div>

      <div v-else class="sidebar-body">
        <div class="novel-title-row">
          <div class="novel-title" :title="novelStore.title">{{ novelStore.title }}</div>
        </div>

        <div class="chapter-stats">
          <div class="stat-pill">
            <el-icon><Files /></el-icon>
            <span>{{ novelStore.chapters.length }} 章</span>
          </div>
          <div class="stat-pill done-pill" v-if="novelStore.chapters.some(c => c.status === 'done')">
            <span class="stat-dot done"></span>
            <span>{{ novelStore.chapters.filter(c => c.status === 'done').length }} 已完成</span>
          </div>
        </div>

        <div class="chapter-list">
          <transition-group name="slide-up" tag="div">
            <div
              v-for="c in novelStore.chapters"
              :key="c.id"
              class="chapter-item"
              :class="{
                active: c.id === novelStore.activeChapterId,
                done: c.status === 'done',
                running: c.status === 'running',
                queued: c.status === 'queued',
                error: c.status === 'error',
                canceled: c.status === 'canceled',
              }"
              @click="novelStore.setActive(c.id)"
            >
              <div class="chapter-title" @dblclick.stop="startEditChapterTitle(c)">
                <span
                  class="status-dot"
                  :class="c.status || 'idle'"
                ></span>
                <input
                  v-if="editingChapterId === c.id"
                  v-model="editingChapterTitle"
                  class="chapter-title-input"
                  ref="chapterTitleInput"
                  @click.stop
                  @keydown.enter="commitChapterTitle(c)"
                  @keydown.esc="cancelChapterTitle"
                  @blur="commitChapterTitle(c)"
                  maxlength="50"
                />
                <span v-else :title="'双击重命名'">{{ c.title }}</span>
              </div>
              <div class="chapter-meta">
                <span class="char-count">{{ c.content.length }}字</span>
                <span v-if="chapterStatusLabel(c)" class="job-label">
                  {{ chapterStatusLabel(c) }}
                </span>
              </div>
            </div>
          </transition-group>
        </div>

        <div class="batch-area">
          <el-button
            v-if="!currentNovelQueueRunning"
            type="primary"
            plain
            :disabled="currentRewriteStartDisabled || !novelStore.chapters.length"
            @click="rewriteAll"
            class="full-btn"
          >
            <el-icon><MagicStick /></el-icon>
            <span style="margin-left: 4px">整本洗稿</span>
          </el-button>
          <div v-else class="queue-status">
            <el-progress
              :percentage="queueProgress.total ? (queueCompleted / queueProgress.total) * 100 : 0"
              :show-text="false"
              :stroke-width="6"
            />
            <div class="queue-activity">
              <span class="activity-spinner" aria-hidden="true"></span>
              <span class="queue-stage">{{ queueStageText }}</span>
              <span v-if="queueEtaText" class="queue-eta">{{ queueEtaText }}</span>
            </div>
            <div class="queue-line">
              <span class="shimmer-text">{{ queueCompleted }} / {{ queueProgress.total }} 章</span>
              <el-button size="small" link type="danger" @click="stopQueue">
                停止
              </el-button>
            </div>
            <div v-if="queueProgress.failed > 0" class="queue-failed">
              ⚠ 已失败 {{ queueProgress.failed }} 章（不会被覆盖）
            </div>
          </div>
          <div v-if="atParallelNovelLimit" class="queue-failed">
            最多同时洗 {{ MAX_PARALLEL_REWRITE_NOVELS }} 本小说
          </div>
          <!-- Retry failed chapters surfaces only after a batch finishes
               with errors. Hidden during normal flow to keep the sidebar
               uncluttered. -->
          <el-button
            v-if="!currentNovelQueueRunning && failedCount > 0"
            type="warning"
            plain
            @click="retryFailedChapters"
            class="full-btn"
          >
            <el-icon><RefreshRight /></el-icon>
            <span style="margin-left: 4px">重试失败的 {{ failedCount }} 章</span>
          </el-button>
          <el-dropdown
            class="full-btn export-dropdown"
            trigger="click"
            :disabled="exportDisabled"
            @command="handleExportCommand"
          >
            <el-button
              plain
              :disabled="exportDisabled"
              class="full-btn export-btn"
            >
              <el-icon><Download /></el-icon>
              <span style="margin-left: 4px">导出文稿</span>
              <el-icon class="export-arrow"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="source" :disabled="!canExportSource">导出原稿</el-dropdown-item>
                <el-dropdown-item command="base" :disabled="!canExportBase">导出洗稿</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </div>
    </aside>

    <!-- Main workspace -->
    <section class="main">
      <div class="control-bar">
        <div
          v-for="context in otherRewriteContexts"
          :key="context.novelId"
          class="background-rewrite-status"
        >
          <span class="activity-spinner" aria-hidden="true"></span>
          <span :title="backgroundRewriteText(context)">{{ backgroundRewriteText(context) }}</span>
          <el-button size="small" link type="danger" @click="stopActiveRewrite(context)">
            停止
          </el-button>
        </div>

        <div class="model-switcher">
          <el-select
            v-if="modelStore.models.length"
            :model-value="modelStore.activeId"
            size="default"
            placeholder="选择模型"
            class="model-switcher-select"
            @update:model-value="onSwitchModel"
          >
            <template #prefix>
              <el-icon><MagicStick /></el-icon>
            </template>
            <el-option
              v-for="m in modelStore.models"
              :key="m.id"
              :value="m.id"
              :label="m.name"
            >
              <div class="model-switcher-opt">
                <span class="model-switcher-opt-name">{{ m.name }}</span>
                <span class="model-switcher-opt-id">{{ m.model }}</span>
              </div>
            </el-option>
          </el-select>
          <router-link v-else to="/settings">
            <el-button size="default" type="primary" plain>
              <el-icon><MagicStick /></el-icon>
              <span style="margin-left: 4px">去配置模型</span>
            </el-button>
          </router-link>
        </div>

        <div style="flex: 1"></div>

        <transition name="fade" mode="out-in">
          <el-button
            v-if="currentNovelQueueRunning"
            key="batch"
            type="danger"
            size="default"
            @click="stopQueue"
            class="cta-btn"
            title="Esc 或 Cmd/Ctrl+Enter 也可停止"
          >
            <el-icon><CircleClose /></el-icon>
            <span style="margin-left: 4px">
              停止批量 · {{ queueCompleted }}/{{ queueProgress.total }}
            </span>
          </el-button>
          <el-button
            v-else-if="!running || !isCurrentNovelStreaming"
            key="start"
            type="primary"
            size="default"
            :disabled="currentRewriteStartDisabled || !activeChapter || !hasUsableModel"
            @click="startRewrite"
            class="cta-btn"
            title="快捷键 Cmd/Ctrl+Enter"
          >
            <el-icon>
              <RefreshRight v-if="hasBaseRewrite" />
              <MagicStick v-else />
            </el-icon>
            <span style="margin-left: 4px">
              {{ hasBaseRewrite ? '重新洗稿' : '开始洗稿' }}
            </span>
          </el-button>
          <el-button
            v-else
            key="stop"
            type="danger"
            size="default"
            @click="stopRewrite"
            class="cta-btn"
            title="Esc 或 Cmd/Ctrl+Enter"
          >
            <el-icon><CircleClose /></el-icon>
            <span style="margin-left: 4px">停止生成</span>
          </el-button>
        </transition>
      </div>

      <div class="dual-pane">
        <div class="pane card-panel pane-source">
          <div class="pane-head">
            <div class="pane-head-left">
              <el-icon class="pane-icon"><Document /></el-icon>
              <span class="pane-title">原文</span>
            </div>
            <div class="pane-stat">{{ origLen }} 字</div>
          </div>
          <el-input
            v-if="activeChapter"
            v-model="activeChapterContent"
            type="textarea"
            :autosize="false"
            resize="none"
            placeholder="选中章节后这里显示原文"
            class="pane-text"
          />
          <div v-else class="empty-state">
            <div class="empty-state-icon">
              <el-icon :size="32"><Right /></el-icon>
            </div>
            <div class="empty-state-title">未选中章节</div>
            <div class="empty-state-desc">从左侧导入原稿、选中章节后查看原文</div>
          </div>
        </div>

        <div class="pane card-panel pane-result" :class="{ 'is-running': isCurrentNovelStreaming }">
          <div class="pane-head">
            <div class="pane-head-left">
              <el-icon class="pane-icon"><MagicStick /></el-icon>
              <span class="pane-title">
                {{ paneTitle }}
                <span v-if="running && isCurrentNovelStreaming" class="shimmer-text running-tag">
                  <span class="mini-pulse" aria-hidden="true"></span>
                  {{ rewriteActivityText }}
                </span>
                <span v-else-if="currentNovelQueueRunning" class="shimmer-text running-tag">
                  <span class="mini-pulse" aria-hidden="true"></span>
                  批量中 · {{ queueCompleted }}/{{ queueProgress.total }}
                  <template v-if="queueEtaText"> · {{ queueEtaText }}</template>
                </span>
              </span>
            </div>
            <div class="pane-actions">
              <el-button
                size="small"
                link
                :disabled="!currentRewritten || isCurrentNovelStreaming"
                @click="copyRewritten"
                :class="{ 'copy-just-copied': justCopied }"
                :title="viewVariant === 'source' ? '复制原稿' : '复制洗稿'"
              >
                <el-icon>
                  <Check v-if="justCopied" />
                  <DocumentCopy v-else />
                </el-icon>
                <span style="margin-left: 4px">
                  {{ justCopied ? '已复制' : '复制' }}
                </span>
              </el-button>
            </div>
          </div>
          <div class="pane-text result-text" ref="resultRef">
            <template v-if="currentRewritten">
              <span class="result-body">{{ currentRewritten }}</span>
              <span v-if="isCurrentNovelStreaming && viewVariant !== 'source'" class="stream-cursor"></span>
            </template>
            <div v-else-if="isCurrentNovelStreaming" class="result-loading">
              <div class="result-loading-bar"></div>
              <div class="result-loading-bar short"></div>
              <div class="result-loading-bar"></div>
              <div class="result-loading-text shimmer-text">
                <span class="activity-spinner" aria-hidden="true"></span>
                {{ rewriteActivityText }}
              </div>
            </div>
            <div v-else class="empty-state">
              <div class="empty-state-icon">
                <el-icon :size="32"><MagicStick /></el-icon>
              </div>
              <div class="empty-state-title">尚未洗稿</div>
              <div class="empty-state-desc">点击「开始洗稿」，完成后结果会自动显示在这里</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Import dialog -->
    <el-dialog v-model="importVisible" title="导入小说" width="640px" top="6vh">
      <div
        class="import-dropzone"
        :class="{ 'is-over': importDragOver }"
        @dragenter="onImportDragEnter"
        @dragover.prevent
        @dragleave="onImportDragLeave"
        @drop="onImportDrop"
      >
        <el-form label-position="top">
          <el-form-item label="小说名称">
            <el-input
              v-model="importTitle"
              placeholder="例：穿越后我捡了个女儿"
              maxlength="60"
              show-word-limit
            />
          </el-form-item>
          <el-form-item :label="`原稿内容（${importCharCount} / ${MAX_NOVEL_CHARS} 字）`">
            <el-alert
              v-if="evalCorpusSummary"
              class="import-corpus-alert"
              type="success"
              :closable="false"
              show-icon
              :title="`测试资料包：${evalCorpusSummary.pair_count || 0} 组原稿/精修，参考重合中位数 ${((evalCorpusSummary.reference_quality?.overlap_median || 0) * 100).toFixed(1)}%`"
            />
            <div class="import-actions">
              <el-button :loading="loadingDocx" @click="pickFile">
                <el-icon><Upload /></el-icon>
                <span style="margin-left: 4px">选择 TXT / DOCX / ZIP 文件</span>
              </el-button>
              <input
                ref="fileInput"
                type="file"
                accept=".txt,.docx,.zip"
                style="display: none"
                @change="onFileChange"
              />
              <span class="hint">TXT / DOCX 是小说导入；ZIP 会作为内部测试资料包导入评测集</span>
            </div>
            <el-input
              v-model="importText"
              type="textarea"
              :rows="14"
              :maxlength="MAX_NOVEL_CHARS"
              show-word-limit
              placeholder="原稿内容粘贴在此..."
            />
          </el-form-item>
        </el-form>
        <div v-if="importDragOver" class="dropzone-overlay">
          <el-icon :size="40"><Upload /></el-icon>
          <div style="margin-top: 10px; font-size: 15px; font-weight: 600">
            松开导入 TXT / DOCX / ZIP
          </div>
        </div>
      </div>
      <template #footer>
        <el-button @click="importVisible = false">取消</el-button>
        <el-button type="primary" :loading="importSubmitting" @click="confirmImport">
          <el-icon><Check /></el-icon>
          <span style="margin-left: 4px">导入并拆分</span>
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.workbench {
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

/* ===== Sidebar ===== */
.sidebar {
  width: 290px;
  flex-shrink: 0;
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 18px 16px;
  overflow: hidden;
}

.sidebar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.novel-picker {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
}

.novel-select {
  flex: 1;
}

.novel-select :deep(.el-select__wrapper) {
  border-radius: 10px !important;
  background: var(--panel-2) !important;
}

.novel-opt {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 100%;
  padding: 2px 0;
}

.novel-opt-title {
  font-weight: 500;
  font-size: 13px;
}

.novel-opt-meta {
  font-size: 11px;
  color: var(--text-mute);
}

.save-status {
  font-size: 11px;
  color: var(--text-mute);
  margin-bottom: 12px;
  height: 16px;
  display: flex;
  align-items: center;
}

.save-status-saving,
.save-status-ok {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.save-status-ok {
  color: var(--success);
}

.save-status-saving {
  color: var(--accent);
}

.save-status-err {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--danger);
}

/* Brief "✓ copied" state on the copy button — replaces the icon and label
 * for ~1.8s so users get unmistakable feedback. */
.copy-just-copied {
  color: var(--success) !important;
}

.rotate {
  animation: rotate 1s linear infinite;
}

@keyframes rotate {
  to { transform: rotate(360deg); }
}

.sidebar-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.novel-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}

.novel-title {
  flex: 1;
  font-weight: 600;
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chapter-stats {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.stat-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  font-size: 11px;
  color: var(--text-soft);
  font-weight: 500;
}

.stat-pill .el-icon {
  font-size: 12px;
}

.stat-pill.done-pill {
  color: var(--success);
  background: var(--success-soft);
  border-color: rgba(16, 185, 129, 0.28);
}

.stat-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
.stat-dot.done { background: var(--success); }

.chapter-list {
  flex: 1;
  overflow-y: auto;
  margin: 0 -4px;
  padding: 0 4px 6px;
}

.chapter-item {
  padding: 9px 11px;
  margin-bottom: 4px;
  border-radius: 8px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: background var(--duration-fast) var(--ease),
              border-color var(--duration-fast) var(--ease),
              transform var(--duration-fast) var(--ease);
}

.chapter-item:hover {
  background: var(--panel-2);
  transform: translateX(2px);
}

.chapter-item.active {
  background: var(--accent-soft);
  border-color: var(--accent);
  box-shadow: 0 2px 8px var(--accent-glow);
}

.chapter-item.running {
  background: var(--accent-soft);
}

.chapter-item.queued {
  background: var(--panel-2);
}

.chapter-title {
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 6px;
}

.chapter-title-input {
  flex: 1;
  font: inherit;
  border: 1px solid var(--accent);
  background: var(--panel);
  border-radius: 4px;
  padding: 1px 6px;
  outline: none;
  width: 100%;
  min-width: 0;
}

.chapter-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-mute);
  padding-left: 14px;
}

.char-count {
  font-variant-numeric: tabular-nums;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--border-strong);
  transition: background var(--duration-fast) var(--ease);
}

.status-dot.running {
  background: var(--accent);
  animation: pulse-soft 1s infinite ease-in-out;
}

.status-dot.queued {
  background: var(--warning, #f59e0b);
}

.status-dot.done {
  background: var(--success);
}

.status-dot.error {
  background: var(--danger);
}

.status-dot.canceled {
  background: var(--text-mute);
}

.job-label {
  color: var(--text-mute);
}

.batch-area {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.full-btn {
  width: 100%;
}

.export-btn {
  font-weight: 500;
}

.export-dropdown {
  display: block;
}

.export-arrow {
  margin-left: auto;
  font-size: 12px;
}

.queue-status {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  background: var(--accent-soft);
  border-radius: 10px;
}

.queue-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: var(--text-soft);
  font-weight: 600;
}

.queue-activity {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  font-size: 12px;
  font-weight: 700;
  color: var(--text);
}

.activity-spinner {
  width: 14px;
  height: 14px;
  flex: 0 0 auto;
  border: 2px solid rgba(99, 102, 241, 0.22);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}

.queue-stage {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.queue-eta {
  flex: 0 0 auto;
  color: var(--text-soft);
  font-weight: 600;
  white-space: nowrap;
}

.queue-failed {
  font-size: 11px;
  color: var(--danger);
  font-weight: 500;
}

/* ===== Main pane ===== */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 14px;
  overflow: hidden;
}

.control-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-xs);
  animation: fade-up var(--duration-slow) var(--ease-out);
}

.model-switcher {
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
}

.model-switcher-select {
  width: 200px;
}

.model-switcher-opt {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
}

.model-switcher-opt-name {
  font-weight: 500;
  font-size: 13px;
}

.model-switcher-opt-id {
  font-size: 11px;
  color: var(--text-mute);
  font-family: ui-monospace, Consolas, monospace;
}

.background-rewrite-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  max-width: min(520px, 58vw);
  padding: 5px 10px;
  border: 1px solid rgba(99, 102, 241, 0.24);
  border-radius: 8px;
  background: var(--accent-soft);
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
}

.background-rewrite-status > span:not(.activity-spinner) {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.cta-btn {
  min-width: 132px;
  height: 36px;
  font-size: 14px;
  font-weight: 600;
}

.dual-pane {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  overflow: hidden;
  min-height: 0;
}

.pane {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: fade-up var(--duration-slow) var(--ease-out);
}

.pane-result.is-running {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-glow), var(--shadow-sm);
}

.pane-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}

.pane-head-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pane-icon {
  color: var(--accent);
  font-size: 15px;
}

.pane-title {
  font-weight: 600;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: 10px;
}

.running-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: min(280px, 42vw);
  min-width: 0;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  padding: 1px 8px;
  border-radius: var(--radius-pill);
  background: var(--accent-soft);
  border: 1px solid rgba(99, 102, 241, 0.3);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.mini-pulse {
  width: 6px;
  height: 6px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 0 var(--accent-glow);
  animation: pulse-dot 1.3s ease-in-out infinite;
}

.pane-stat {
  font-size: 12px;
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

.pane-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.pane-text {
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

.pane-text :deep(.el-textarea__inner) {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  height: 100% !important;
  resize: none !important;
  font-size: 14px;
  line-height: 1.85;
  padding: 16px 18px;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  color: var(--text);
}

.result-text {
  flex: 1;
  overflow-y: auto;
  padding: 16px 18px;
  font-size: 14px;
  line-height: 1.85;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: var(--text);
  scroll-behavior: smooth;
}

.result-body {
  display: inline;
}

.result-loading {
  padding: 24px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.result-loading-bar {
  height: 12px;
  background: linear-gradient(
    90deg,
    var(--panel-3) 0%,
    var(--accent-soft) 50%,
    var(--panel-3) 100%
  );
  background-size: 200% 100%;
  border-radius: 6px;
  animation: shimmer 1.6s linear infinite;
}

.result-loading-bar.short {
  width: 65%;
}

.result-loading-text {
  margin-top: 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  text-align: center;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes pulse-dot {
  0% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.35);
  }
  70% {
    box-shadow: 0 0 0 7px rgba(99, 102, 241, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0);
  }
}

/* ===== Empty states ===== */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
  color: var(--text-mute);
}

.empty-state-icon {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: var(--panel-2);
  display: grid;
  place-items: center;
  color: var(--accent);
  margin-bottom: 14px;
  box-shadow: var(--shadow-xs);
}

.empty-state-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-soft);
  margin-bottom: 4px;
}

.empty-state-desc {
  font-size: 12px;
  color: var(--text-mute);
  line-height: 1.6;
}

.import-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.import-corpus-alert {
  margin-bottom: 12px;
}

.hint {
  color: var(--text-mute);
  font-size: 12px;
}

.import-dropzone {
  position: relative;
  border-radius: 10px;
  transition: background var(--duration-fast) var(--ease);
}

.import-dropzone.is-over {
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 2px var(--accent);
}

.dropzone-overlay {
  position: absolute;
  inset: 0;
  background: rgba(99, 102, 241, 0.85);
  color: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  pointer-events: none;
}

/* === Narrow screen adaptation === */

/* Tablet / small laptop: shrink the sidebar but keep three-column layout. */
@media (max-width: 1180px) {
  .sidebar {
    width: 240px;
    padding: 14px 12px;
  }
  .control-bar {
    flex-wrap: wrap;
    gap: 10px;
  }
}

/* Phone-ish: stack vertically. Sidebar collapses to a top strip with the
 * novel-picker only; chapter list moves into a horizontally scrollable row;
 * the two panes stack. */
@media (max-width: 820px) {
  .workbench {
    flex-direction: column;
    overflow-y: auto;
  }
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 12px;
  }
  .chapter-list {
    display: flex;
    overflow-x: auto;
    overflow-y: visible;
    gap: 6px;
    padding-bottom: 6px;
    margin: 0;
  }
  .chapter-item {
    flex: 0 0 auto;
    min-width: 150px;
    max-width: 200px;
  }
  .main {
    padding: 12px;
  }
  .background-rewrite-status {
    width: 100%;
    max-width: none;
  }
  .dual-pane {
    grid-template-columns: 1fr;
  }
  .pane {
    min-height: 320px;
  }
}
</style>
