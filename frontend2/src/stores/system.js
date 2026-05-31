import { defineStore } from 'pinia'
import { api } from '../api/v2'

export const useSystemStore = defineStore('system', {
  state: () => ({
    max_concurrency: 5,
    max_chapter_size: 2200,
    loading: false,
  }),
  actions: {
    async load() {
      this.loading = true
      try {
        const cfg = await api.getSystem()
        Object.assign(this, cfg)
      } finally {
        this.loading = false
      }
    },
    async save(payload) {
      const cfg = await api.setSystem(payload)
      Object.assign(this, cfg)
    },
  },
})
