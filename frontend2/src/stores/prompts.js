import { defineStore } from 'pinia'
import { api } from '../api/v2'

export const usePromptStore = defineStore('prompts', {
  state: () => ({
    prompts: [],
    activeId: localStorage.getItem('active_prompt_id') || 'builtin:洗稿',
    activeScriptId: localStorage.getItem('active_script_prompt_id') || 'builtin:转剧本',
  }),
  getters: {
    activePrompt(state) {
      return (
        state.prompts.find((p) => p.id === state.activeId) ||
        state.prompts[0] ||
        null
      )
    },
  },
  actions: {
    async loadAll() {
      this.prompts = await api.listPrompts()
      const aliases = {
        'builtin:精修': 'builtin:洗稿',
        'builtin:洗稿剧本版': 'builtin:转剧本',
        'builtin:精修剧本版': 'builtin:转剧本',
      }
      if (aliases[this.activeId]) {
        this.activeId = aliases[this.activeId]
      }
      if (aliases[this.activeScriptId]) {
        this.activeScriptId = aliases[this.activeScriptId]
      }
      const rewritePrompts = this.prompts.filter((p) => p.task !== 'script')
      const scriptPrompts = this.prompts.filter((p) => p.task === 'script')
      const active = this.prompts.find((p) => p.id === this.activeId)
      if ((!active || active.task === 'script') && rewritePrompts[0]) {
        this.activeId = rewritePrompts[0].id
      } else if (!active && this.prompts[0]) {
        this.activeId = this.prompts[0].id
      }
      const activeScript = this.prompts.find((p) => p.id === this.activeScriptId)
      if ((!activeScript || activeScript.task !== 'script') && scriptPrompts[0]) {
        this.activeScriptId = scriptPrompts[0].id
      }
      localStorage.setItem('active_prompt_id', this.activeId)
      localStorage.setItem('active_script_prompt_id', this.activeScriptId)
    },
    setActive(id) {
      this.activeId = id
      localStorage.setItem('active_prompt_id', id)
    },
    setActiveScript(id) {
      this.activeScriptId = id
      localStorage.setItem('active_script_prompt_id', id)
    },
    async save(payload) {
      const saved = await api.upsertPrompt(payload)
      await this.loadAll()
      this.setActive(saved.id)
      return saved
    },
    async remove(id) {
      await api.deletePrompt(id)
      await this.loadAll()
    },
  },
})
