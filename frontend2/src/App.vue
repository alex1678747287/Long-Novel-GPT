<script setup>
import { onMounted } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { useModelStore } from './stores/models'
import { usePromptStore } from './stores/prompts'
import { useSystemStore } from './stores/system'
import { useNovelStore } from './stores/novel'

const modelStore = useModelStore()
const promptStore = usePromptStore()
const systemStore = useSystemStore()
const novelStore = useNovelStore()

onMounted(async () => {
  try {
    await Promise.all([
      modelStore.loadAll(),
      promptStore.loadAll(),
      systemStore.load(),
      novelStore.loadAll(),
    ])
  } catch (e) {
    console.error('initial load failed', e)
  }
})

// Flush any in-flight debounced edits when the tab unloads so we don't lose
// the last keystroke.
window.addEventListener('beforeunload', () => {
  try {
    novelStore.flushPendingEdits()
  } catch {}
})

</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div class="brand">
        <div class="brand-logo">精</div>
        <div class="brand-title">
          <span>精修工作台</span>
          <span class="sub">短剧精修引擎</span>
        </div>
      </div>
      <div class="spacer"></div>
      <nav class="nav">
        <RouterLink to="/" class="nav-item" exact-active-class="active">工作台</RouterLink>
        <RouterLink to="/settings" class="nav-item" exact-active-class="active">模型配置</RouterLink>
      </nav>
    </header>

    <main class="app-main">
      <RouterView v-slot="{ Component }">
        <transition name="slide-up" mode="out-in">
          <component :is="Component" />
        </transition>
      </RouterView>
    </main>
  </div>
</template>
