import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'workbench', component: () => import('../views/Workbench.vue') },
  { path: '/settings', name: 'settings', component: () => import('../views/Settings.vue') },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
})
