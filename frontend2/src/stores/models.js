import { defineStore } from 'pinia'
import { api } from '../api/v2'

export const useModelStore = defineStore('models', {
  state: () => ({
    models: [],
    presets: [],
    activeId: null,
    loading: false,
  }),
  getters: {
    activeModel(state) {
      return state.models.find((m) => m.id === state.activeId) || null
    },
  },
  actions: {
    async loadAll() {
      this.loading = true
      try {
        const [presetData, modelData] = await Promise.all([
          api.presets(),
          api.listModels(),
        ])
        this.presets = presetData
        this.models = modelData.models
        this.activeId = this.models.some((m) => m.id === modelData.active_id)
          ? modelData.active_id
          : null
      } finally {
        this.loading = false
      }
    },
    async save(payload) {
      const saved = await api.upsertModel(payload)
      await this.loadAll()
      return saved
    },
    async remove(id) {
      await api.deleteModel(id)
      await this.loadAll()
    },
    async activate(id) {
      await api.activateModel(id)
      this.activeId = id
    },
    async test(id) {
      return api.testModel(id)
    },
  },
})
