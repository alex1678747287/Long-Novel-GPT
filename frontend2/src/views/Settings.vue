<script setup>
import { ref, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Key, Link, Check, Star, Download, Upload } from '@element-plus/icons-vue'
import { useModelStore } from '../stores/models'
import { useNovelStore } from '../stores/novel'
import { api } from '../api/v2'

const modelStore = useModelStore()
const novelStore = useNovelStore()

// ---- Quick-add (preset-driven) ----
const quickAddVisible = ref(false)
const quickAddPreset = ref(null)
const showQuickAddAdvanced = ref(false)
const quickAddForm = ref({ id: '', name: '', api_key: '', model: '', base_url: '', temperature: 0.7, max_tokens: 8192, note: '' })

function defaultMaxTokensForPreset(preset) {
  if (['apimart', 'openrouter', 'openai', 'deepseek'].includes(preset?.id)) {
    return 16384
  }
  return 8192
}

function openQuickAdd(preset) {
  quickAddPreset.value = preset
  showQuickAddAdvanced.value = false
  quickAddForm.value = {
    id: '',
    name: preset.label,
    api_key: '',
    model: preset.default_model,
    base_url: preset.base_url,
    temperature: 0.7,
    max_tokens: defaultMaxTokensForPreset(preset),
    note: '',
  }
  quickAddVisible.value = true
}

function openEdit(m) {
  const preset =
    modelStore.presets.find((p) => p.id === m.preset_id) ||
    quickAddPreset.value ||
    modelStore.presets[0] ||
    { id: m.preset_id || 'custom', label: m.name, base_url: m.base_url, default_model: m.model, models: [] }
  quickAddPreset.value = preset
  showQuickAddAdvanced.value = true
  quickAddForm.value = {
    id: m.id,
    name: m.name,
    api_key: '',
    model: m.model,
    base_url: m.base_url ?? preset.base_url,
    temperature: m.temperature ?? 0.7,
    max_tokens: m.max_tokens ?? defaultMaxTokensForPreset(preset),
    note: m.note ?? '',
  }
  quickAddVisible.value = true
}

async function saveQuickAdd() {
  const preset = quickAddPreset.value
  const editId = quickAddForm.value.id || ''
  const isEdit = !!editId
  // New models require an API key; edits may keep the existing key by leaving it blank.
  if (!isEdit && !quickAddForm.value.api_key.trim()) {
    ElMessage.error('请填入 API Key')
    return
  }
  try {
    const wasActive = isEdit && editId === modelStore.activeId
    const saved = await modelStore.save({
      id: editId,
      name: quickAddForm.value.name.trim() || preset.label,
      preset_id: preset.id,
      base_url: quickAddForm.value.base_url || preset.base_url,
      api_key: quickAddForm.value.api_key.trim(),
      model: quickAddForm.value.model.trim() || preset.default_model,
      temperature: quickAddForm.value.temperature ?? 0.7,
      max_tokens: quickAddForm.value.max_tokens ?? defaultMaxTokensForPreset(preset),
      note: quickAddForm.value.note || '',
    })
    quickAddVisible.value = false
    if (isEdit) {
      ElMessage.success(`已保存 ${saved.name || quickAddForm.value.name}`)
      // Only re-activate if the edited card was the active model, to avoid
      // silently switching the active model when editing an inactive one.
      if (wasActive) await modelStore.activate(saved.id)
    } else {
      ElMessage.success(`已添加 ${preset.label}`)
      await modelStore.activate(saved.id)
    }
  } catch (e) {
    ElMessage.error('保存失败：' + e.message)
  }
}

// Returns the number of models already added for this preset.
function presetCount(presetId) {
  return modelStore.models.filter((m) => m.preset_id === presetId).length
}

async function removeModel(m) {
  try {
    await ElMessageBox.confirm(`删除模型「${m.name}」？`, '确认删除', { type: 'warning' })
  } catch { return }
  await modelStore.remove(m.id)
  ElMessage.success('已删除')
}

const testingId = ref(null)
async function testModel(m) {
  testingId.value = m.id
  try {
    const res = await modelStore.test(m.id)
    if (res.ok) ElMessage.success(`连接成功 · ${res.reply.slice(0, 30)}`)
    else ElMessage.error('连接失败：' + res.error)
  } catch (e) {
    ElMessage.error('测试失败：' + e.message)
  } finally {
    testingId.value = null
  }
}

async function activate(m) {
  await modelStore.activate(m.id)
  ElMessage.success(`已激活：${m.name}`)
}

// ---- Backup / restore ----
const backupFileInput = ref(null)

async function doExportBackup() {
  try {
    const blob = await api.exportBackup()
    const json = JSON.stringify(blob, null, 2)
    const file = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(file)
    const a = document.createElement('a')
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
    a.href = url
    a.download = `xigao_backup_${ts}.json`
    a.click()
    URL.revokeObjectURL(url)
    const novels = blob.novels?.length || 0
    const chapters = blob.chapters?.length || 0
    ElMessage.success(`已导出 ${novels} 本小说 / ${chapters} 章`)
  } catch (e) {
    ElMessage.error('导出失败：' + e.message)
  }
}

function pickBackupFile() {
  backupFileInput.value?.click()
}

async function onBackupFileChosen(e) {
  const file = e.target.files?.[0]
  e.target.value = ''
  if (!file) return
  let data
  try {
    const text = await file.text()
    data = JSON.parse(text)
  } catch (err) {
    ElMessage.error('文件格式错误：' + err.message)
    return
  }
  let merge = true
  try {
    await ElMessageBox.confirm(
      `检测到备份文件，包含 ${data.novels?.length || 0} 本小说 / ${data.chapters?.length || 0} 章。\n\n` +
        '选择导入模式：\n' +
        '· 合并（推荐）：只追加现有库里没有的小说\n' +
        '· 覆盖：先清空当前所有数据，再导入',
      '导入备份',
      {
        type: 'warning',
        confirmButtonText: '合并导入',
        cancelButtonText: '覆盖恢复',
        distinguishCancelAndClose: true,
        showClose: true,
      }
    )
    merge = true
  } catch (action) {
    if (action !== 'cancel') return
    try {
      await ElMessageBox.confirm(
        '覆盖恢复会先清空当前所有小说、章节和洗稿结果，再导入备份。此操作不可撤销，确定继续？',
        '确认覆盖恢复',
        { type: 'error', confirmButtonText: '确认覆盖恢复', cancelButtonText: '返回' }
      )
      merge = false
    } catch {
      return
    }
  }
  try {
    const res = await api.importBackup(data, merge)
    ElMessage.success(`导入完成，新增 ${res.inserted} 本小说`)
    await novelStore.loadAll()
  } catch (e) {
    ElMessage.error('导入失败：' + e.message)
  }
}

// Visual helpers for presets
const presetIcons = {
  doubao: '🫘',
  zhipuai: '🧠',
  deepseek: '🔍',
  moonshot: '🌙',
  qwen: '☁️',
  openai: '✦',
  openrouter: '🔀',
  apimart: '🛒',
}
const usablePresets = computed(() => modelStore.presets.filter((p) => p.id !== 'custom'))
</script>

<template>
  <div class="settings-page">
    <div class="settings-container">
      <div class="settings-panel">
        <div class="section-bar">
          <div>
            <h3>选择 AI 服务商</h3>
            <p class="hint">点击下方任一服务商，只需粘贴 API Key 即可一键完成对接。所有凭据本地存储，不会上传。</p>
          </div>
        </div>

        <!-- Preset cards (one-click add) -->
        <div class="preset-grid">
          <div
            v-for="p in usablePresets"
            :key="p.id"
            class="preset-card"
            :class="{ configured: presetCount(p.id) > 0 }"
            @click="openQuickAdd(p)"
          >
            <div class="preset-icon">{{ presetIcons[p.id] || '✨' }}</div>
            <div class="preset-info">
              <div class="preset-name">
                {{ p.label }}
                <el-tag v-if="presetCount(p.id) > 0" size="small" type="success" effect="light">
                  已配置 {{ presetCount(p.id) }}
                </el-tag>
              </div>
              <div class="preset-desc">粘贴密钥即可使用</div>
            </div>
            <div class="preset-cta">
              <el-icon><Plus /></el-icon>
            </div>
          </div>
        </div>

        <!-- Configured models -->
        <div v-if="modelStore.models.length" class="configured-section">
          <div class="section-bar">
            <div>
              <h3>已对接的模型</h3>
              <p class="hint">点击"激活"切换当前使用的模型；点击"测试"验证连通性。</p>
            </div>
          </div>

          <div class="model-grid">
            <div
              v-for="m in modelStore.models"
              :key="m.id"
              class="model-card card-panel"
              :class="{ active: m.id === modelStore.activeId }"
            >
              <div class="model-card-head">
                <div class="model-card-title">
                  <div class="model-name">
                    <span class="model-emoji">{{ presetIcons[m.preset_id] || '✦' }}</span>
                    {{ m.name }}
                  </div>
                  <el-tag
                    v-if="m.id === modelStore.activeId"
                    type="success"
                    size="small"
                    effect="light"
                  >
                    <el-icon><Star /></el-icon>
                    <span style="margin-left: 2px">当前</span>
                  </el-tag>
                </div>
              </div>
              <div class="model-actions">
                <el-button size="small" :type="m.id === modelStore.activeId ? 'success' : 'primary'" :disabled="m.id === modelStore.activeId" @click="activate(m)">
                  {{ m.id === modelStore.activeId ? '已激活' : '激活' }}
                </el-button>
                <el-button size="small" :loading="testingId === m.id" @click="testModel(m)">测试</el-button>
                <el-button size="small" @click="openEdit(m)">编辑</el-button>
                <el-button size="small" type="danger" plain @click="removeModel(m)">
                  删除
                </el-button>
              </div>
            </div>
          </div>
        </div>

        <div v-else class="empty-state-large">
          <div class="empty-state-icon-large">🔌</div>
          <div class="empty-state-title">还没对接任何模型</div>
          <div class="empty-state-desc">点击上方任一服务商，粘贴密钥即可开始使用</div>
        </div>

        <div class="backup-section">
          <div class="section-bar compact">
            <div>
              <h3>数据备份</h3>
              <p class="hint">导出所有小说和洗稿结果，换机器或重装后可恢复。</p>
            </div>
          </div>
          <div class="backup-actions">
            <el-button type="primary" plain @click="doExportBackup">
              <el-icon><Download /></el-icon>
              <span style="margin-left: 4px">导出备份</span>
            </el-button>
            <el-button plain @click="pickBackupFile">
              <el-icon><Upload /></el-icon>
              <span style="margin-left: 4px">导入备份</span>
            </el-button>
            <input
              ref="backupFileInput"
              type="file"
              accept=".json"
              style="display: none"
              @change="onBackupFileChosen"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Quick-add dialog (only API key needed) -->
    <el-dialog
      v-model="quickAddVisible"
      :title="quickAddForm.id ? '编辑模型' : quickAddPreset ? `对接 ${quickAddPreset.label}` : '添加模型'"
      width="500px"
    >
      <div v-if="quickAddPreset" class="quick-add-body">
        <div class="quick-add-preset">
          <span class="preset-icon-lg">{{ presetIcons[quickAddPreset.id] || '✨' }}</span>
          <div>
            <div class="preset-name-lg">{{ quickAddPreset.label }}</div>
            <a v-if="quickAddPreset.docs" :href="quickAddPreset.docs" target="_blank" class="preset-docs-link">
              <el-icon><Link /></el-icon>
              获取 API Key 文档
            </a>
          </div>
        </div>
        <el-form :model="quickAddForm" label-position="top">
          <el-form-item label="API Key">
            <el-input
              v-model="quickAddForm.api_key"
              type="password"
              show-password
              :placeholder="quickAddForm.id ? '留空则保留原密钥' : (quickAddPreset.id === 'doubao' ? 'ark-...' : 'sk-...')"
              autofocus
            >
              <template #prefix>
                <el-icon><Key /></el-icon>
              </template>
            </el-input>
            <div class="form-hint">
              {{ quickAddForm.id ? '出于安全考虑不显示原密钥；留空则继续使用原密钥，仅在需要更换时填写' : '粘贴你在服务商控制台获取到的 API Key' }}
            </div>
          </el-form-item>
          <el-button class="quick-add-advanced-toggle" link type="primary" @click="showQuickAddAdvanced = !showQuickAddAdvanced">
            {{ showQuickAddAdvanced ? '收起更多模型设置' : '更多模型设置' }}
          </el-button>
          <div v-if="showQuickAddAdvanced" class="quick-add-advanced">
            <el-form-item label="显示名称">
              <el-input v-model="quickAddForm.name" placeholder="例：我的豆包" />
            </el-form-item>
            <el-form-item label="模型 ID">
              <el-select
                v-if="quickAddPreset.models && quickAddPreset.models.length"
                v-model="quickAddForm.model"
                filterable
                allow-create
                default-first-option
                :placeholder="quickAddPreset.default_model"
                style="width: 100%"
              >
                <el-option
                  v-for="m in quickAddPreset.models"
                  :key="m.id"
                  :value="m.id"
                  :label="m.label || m.id"
                >
                  <div class="model-id-opt">
                    <span class="model-id-opt-name">{{ m.label || m.id }}</span>
                    <span class="model-id-opt-id">{{ m.id }}</span>
                  </div>
                </el-option>
              </el-select>
              <el-input
                v-else
                v-model="quickAddForm.model"
                :placeholder="quickAddPreset.default_model"
              />
              <div class="form-hint">
                <template v-if="quickAddPreset.models && quickAddPreset.models.length">
                  从下拉选择 APIMart 支持的模型，或直接输入其他模型 ID
                </template>
                <template v-else>
                  默认使用 {{ quickAddPreset.default_model }}；改为其他同服务商的模型也可以
                </template>
              </div>
            </el-form-item>
          </div>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="quickAddVisible = false">取消</el-button>
        <el-button type="primary" @click="saveQuickAdd">
          <el-icon><Check /></el-icon>
          <span style="margin-left: 4px">{{ quickAddForm.id ? '保存修改' : '添加并激活' }}</span>
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.settings-page {
  width: 100%;
  overflow: auto;
  padding: 24px;
}

.settings-container {
  max-width: 1120px;
  margin: 0 auto;
}

.settings-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 22px 24px 24px;
  box-shadow: var(--shadow-sm);
}

.section-bar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin: 6px 0 18px;
}

.section-bar.compact {
  margin-bottom: 12px;
}

.section-bar h3 {
  margin: 0 0 4px 0;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.1px;
}

.hint {
  color: var(--text-soft);
  font-size: 13px;
  margin: 0;
  line-height: 1.6;
  max-width: 760px;
}

/* ===== Preset cards ===== */
.preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}

.preset-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 16px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: transform var(--duration-fast) var(--ease),
              box-shadow var(--duration-fast) var(--ease),
              border-color var(--duration-fast) var(--ease);
  position: relative;
  overflow: hidden;
}

.preset-card:hover {
  border-color: var(--accent);
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.preset-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--grad-brand);
  opacity: 0;
  transition: opacity var(--duration-fast) var(--ease);
  pointer-events: none;
}

.preset-card:hover::before {
  opacity: 0.03;
}

.preset-card.configured {
  border-color: rgba(16, 185, 129, 0.4);
  background: linear-gradient(to right, var(--success-soft) 0%, var(--panel) 30%);
}

.preset-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: var(--panel-2);
  display: grid;
  place-items: center;
  font-size: 22px;
  flex-shrink: 0;
}

.preset-info {
  flex: 1;
  min-width: 0;
}

.preset-name {
  font-weight: 600;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.preset-desc {
  font-size: 12px;
  color: var(--text-mute);
  line-height: 1.4;
}

.preset-cta {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  display: grid;
  place-items: center;
  font-size: 14px;
  transition: all var(--duration-fast) var(--ease);
}

.preset-card:hover .preset-cta {
  background: var(--accent);
  color: #fff;
  transform: rotate(90deg);
}

/* ===== Quick-add dialog ===== */
.quick-add-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.quick-add-preset {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.preset-icon-lg {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: var(--panel);
  display: grid;
  place-items: center;
  font-size: 24px;
}

.preset-name-lg {
  font-weight: 700;
  font-size: 15px;
}

.preset-docs-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  margin-top: 4px;
  color: var(--accent);
}

.quick-add-advanced-toggle {
  align-self: flex-start;
  padding-left: 0;
}

.quick-add-advanced {
  margin-top: 4px;
  padding: 12px 14px 0;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  background: var(--panel-2);
}

/* ===== Configured models ===== */
.configured-section {
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  margin-top: 8px;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 14px;
}

.model-card {
  padding: 16px 18px;
  transition: border-color var(--duration-fast) var(--ease),
              box-shadow var(--duration-fast) var(--ease),
              transform var(--duration-fast) var(--ease);
}

.model-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.model-card.active {
  border-color: var(--accent);
  background: linear-gradient(160deg, var(--accent-soft) 0%, var(--panel) 50%);
  box-shadow: 0 0 0 2px var(--accent-glow);
}

.model-card-head {
  margin-bottom: 12px;
}

.model-card-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.model-name {
  font-weight: 700;
  font-size: 15px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.model-emoji {
  font-size: 16px;
}

.model-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.backup-section {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px dashed var(--border);
}

.backup-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.form-hint {
  font-size: 12px;
  color: var(--text-mute);
  margin-top: 4px;
}

.model-id-opt {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  padding: 2px 0;
}

.model-id-opt-name {
  font-weight: 500;
  font-size: 13px;
}

.model-id-opt-id {
  font-size: 11px;
  color: var(--text-mute);
  font-family: ui-monospace, Consolas, monospace;
}

.empty-state-large {
  padding: 60px 20px;
  text-align: center;
  color: var(--text-mute);
}

.empty-state-icon-large {
  font-size: 48px;
  margin-bottom: 14px;
}

.empty-state-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-soft);
  margin-bottom: 6px;
}

.empty-state-desc {
  font-size: 13px;
}
</style>
