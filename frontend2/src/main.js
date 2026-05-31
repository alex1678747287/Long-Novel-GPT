import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElIcons from '@element-plus/icons-vue'

import App from './App.vue'
import router from './router'
import './style.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(ElementPlus)

// Register every Element Plus icon as a global component (lazy approach but
// fine for this app's size).
for (const [name, comp] of Object.entries(ElIcons)) {
  app.component(name, comp)
}

app.mount('#app')
