import { showToast } from './utils.js';

let previousMode = null;
let modelConfigs = null;
let apiSettings = null;

const PROVIDER_LABELS = {
    doubao: '豆包 (火山方舟)',
    zhipuai: '智谱 AI',
    gpt: 'OpenAI / 兼容',
    wenxin: '文心一言',
    local: '本地模型',
};

const FIELD_LABELS = {
    api_key: 'API Key',
    base_url: 'Base URL',
    ak: 'AK',
    sk: 'SK',
    endpoint_ids: 'Endpoint IDs (逗号分隔，可填模型ID)',
    available_models: '可用模型 (逗号分隔)',
};

const FIELD_PLACEHOLDERS = {
    doubao: {
        api_key: 'ark-xxxxxxxx',
        endpoint_ids: 'doubao-seed-2-0-pro-260215 或 ep-xxxxx',
        available_models: 'doubao-seed-2.0-pro',
    },
    zhipuai: {
        api_key: 'xxx.xxx',
        available_models: 'glm-4-air,glm-4-flashx',
    },
    gpt: {
        base_url: 'https://api.openai.com/v1',
        api_key: 'sk-...',
        available_models: 'gpt-4o,gpt-4o-mini',
    },
    wenxin: {
        ak: '',
        sk: '',
        available_models: 'ERNIE-Novel-8K,ERNIE-4.0-8K',
    },
    local: {
        base_url: 'http://host.docker.internal:8000/v1',
        api_key: 'local-key',
        available_models: 'local-model-1',
    },
};

function buildApiProvidersHtml(serverApi) {
    const providers = serverApi || {};
    return Object.entries(providers).map(([provider, info]) => {
        const fieldsHtml = (info.fields || []).map(field => {
            const inputType = field.is_secret ? 'password' : 'text';
            const placeholder = (FIELD_PLACEHOLDERS[provider] || {})[field.name] || '';
            return `
                <div class="api-field">
                    <label>${FIELD_LABELS[field.name] || field.name}</label>
                    <input type="${inputType}"
                           data-provider="${provider}"
                           data-field="${field.name}"
                           value="${(field.value || '').toString().replace(/"/g, '&quot;')}"
                           placeholder="${placeholder}"
                           autocomplete="off" />
                </div>`;
        }).join('');

        return `
            <details class="api-provider" data-provider="${provider}">
                <summary>${PROVIDER_LABELS[provider] || provider}</summary>
                <div class="api-provider-body">
                    ${fieldsHtml}
                    <button class="save-api-btn" data-provider="${provider}">保存 ${PROVIDER_LABELS[provider] || provider}</button>
                </div>
            </details>`;
    }).join('');
}

function createSettingsPopup() {
    const overlay = document.createElement('div');
    overlay.className = 'settings-overlay';

    const popup = document.createElement('div');
    popup.className = 'settings-popup';

    const settings = JSON.parse(localStorage.getItem('settings') || '{}');

    popup.innerHTML = `
        <div class="settings-header">
            <div class="header-content">
                <h3>设置</h3>
                <p class="subtitle">配置系统参数、API 凭据和模型选择</p>
            </div>
            <button class="settings-close">&times;</button>
        </div>
        <div class="settings-content">
            <div class="settings-section">
                <h4>系统参数</h4>
                <div class="setting-item">
                    <label for="maxThreadNum">最大线程数</label>
                    <input type="number" id="maxThreadNum" min="1" max="20" value="${settings.MAX_THREAD_NUM || 5}">
                </div>
                <div class="setting-item">
                    <label for="maxNovelSummaryLength">导入小说的最大长度</label>
                    <input type="number" id="maxNovelSummaryLength" min="10000" max="1000000" value="${settings.MAX_NOVEL_SUMMARY_LENGTH || 20000}">
                </div>
            </div>
            <div class="settings-section">
                <h4>API 配置</h4>
                <p class="api-hint">填写后点击对应的保存按钮，凭据会持久化到容器内的 <code>data/user_config.json</code>。</p>
                <div class="api-providers">${buildApiProvidersHtml(apiSettings)}</div>
            </div>
            <div class="settings-section">
                <h4>模型设置</h4>
                <div class="setting-item">
                    <label for="defaultMainModel">主模型</label>
                    <div class="model-select-group">
                        <select id="defaultMainModel"></select>
                        <button class="test-model-btn" data-for="defaultMainModel">测试</button>
                    </div>
                </div>
                <div class="setting-item">
                    <label for="defaultSubModel">辅助模型</label>
                    <div class="model-select-group">
                        <select id="defaultSubModel"></select>
                        <button class="test-model-btn" data-for="defaultSubModel">测试</button>
                    </div>
                </div>
            </div>
        </div>
        <div class="settings-footer">
            <button class="save-settings">保存设置</button>
        </div>
    `;

    overlay.appendChild(popup);
    document.body.appendChild(overlay);

    const closeBtn = popup.querySelector('.settings-close');
    closeBtn.addEventListener('click', hideSettings);

    const saveBtn = popup.querySelector('.save-settings');
    saveBtn.addEventListener('click', () => {
        saveSettings();
        hideSettings();
    });

    popup.querySelectorAll('.save-api-btn').forEach(btn => {
        btn.addEventListener('click', () => saveApiProvider(btn.dataset.provider, btn));
    });

    const testButtons = popup.querySelectorAll('.test-model-btn');
    testButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const selectId = btn.dataset.for;
            const select = document.getElementById(selectId);
            const selectedModel = select.value;

            if (!selectedModel) {
                showToast('请先选择一个模型', 'error');
                return;
            }

            btn.disabled = true;
            btn.textContent = '测试中...';

            try {
                const response = await fetch(`${window._env_?.SERVER_URL}/test_model`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider_model: selectedModel })
                });

                const result = await response.json();

                if (result.success) {
                    showToast('模型测试成功', 'success');
                } else {
                    showToast(`模型测试失败: ${result.error}`, 'error');
                }
            } catch (error) {
                showToast(`测试请求失败: ${error.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '测试';
            }
        });
    });

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            hideSettings();
        }
    });

    return overlay;
}

async function saveApiProvider(provider, btn) {
    const inputs = document.querySelectorAll(`.settings-popup input[data-provider="${provider}"]`);
    const config = {};
    inputs.forEach(input => {
        config[input.dataset.field] = input.value;
    });

    btn.disabled = true;
    const originalText = btn.textContent;
    btn.textContent = '保存中...';

    try {
        const response = await fetch(`${window._env_?.SERVER_URL}/setting/api`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, config })
        });
        const result = await response.json();
        if (result.success) {
            showToast(`${PROVIDER_LABELS[provider] || provider} 配置已保存`, 'success');
            // Refresh model lists in case available_models changed.
            await loadModelConfigs();
            updateModelSelects();
            loadCurrentSettings();
        } else {
            showToast(`保存失败: ${result.error}`, 'error');
        }
    } catch (e) {
        showToast(`保存失败: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

export async function loadModelConfigs() {
    try {
        const response = await fetch(`${window._env_?.SERVER_URL}/setting`);
        const settings = await response.json();
        modelConfigs = settings.models;
        apiSettings = settings.api_settings;

        if (!localStorage.getItem('settings')) {
            localStorage.setItem('settings', JSON.stringify({
                MAIN_MODEL: settings.MAIN_MODEL,
                SUB_MODEL: settings.SUB_MODEL,
                MAX_THREAD_NUM: settings.MAX_THREAD_NUM,
                MAX_NOVEL_SUMMARY_LENGTH: settings.MAX_NOVEL_SUMMARY_LENGTH
            }));
        }
    } catch (error) {
        console.error('Error loading settings:', error);
        showToast('加载设置失败', 'error');
    }
}

function updateModelSelects() {
    const mainModelSelect = document.getElementById('defaultMainModel');
    const subModelSelect = document.getElementById('defaultSubModel');

    if (!mainModelSelect || !subModelSelect || !modelConfigs) return;

    mainModelSelect.innerHTML = '';
    subModelSelect.innerHTML = '';

    Object.entries(modelConfigs).forEach(([provider, models]) => {
        models.forEach(model => {
            if (!model) return;
            const option = document.createElement('option');
            option.value = `${provider}/${model}`;
            option.textContent = `${provider}/${model}`;
            mainModelSelect.appendChild(option.cloneNode(true));
            subModelSelect.appendChild(option.cloneNode(true));
        });
    });
}

function loadCurrentSettings() {
    const settings = JSON.parse(localStorage.getItem('settings') || '{}');
    const mainModelSelect = document.getElementById('defaultMainModel');
    const subModelSelect = document.getElementById('defaultSubModel');

    if (mainModelSelect && mainModelSelect.options.length > 0 && settings.MAIN_MODEL) {
        mainModelSelect.value = settings.MAIN_MODEL;
    }
    if (subModelSelect && subModelSelect.options.length > 0 && settings.SUB_MODEL) {
        subModelSelect.value = settings.SUB_MODEL;
    }

    const threadInput = document.getElementById('maxThreadNum');
    const lengthInput = document.getElementById('maxNovelSummaryLength');
    if (threadInput && settings.MAX_THREAD_NUM) threadInput.value = settings.MAX_THREAD_NUM;
    if (lengthInput && settings.MAX_NOVEL_SUMMARY_LENGTH) lengthInput.value = settings.MAX_NOVEL_SUMMARY_LENGTH;
}

function saveSettings() {
    const settings = {
        MAIN_MODEL: document.getElementById('defaultMainModel').value,
        SUB_MODEL: document.getElementById('defaultSubModel').value,
        MAX_THREAD_NUM: parseInt(document.getElementById('maxThreadNum').value),
        MAX_NOVEL_SUMMARY_LENGTH: parseInt(document.getElementById('maxNovelSummaryLength').value)
    };

    localStorage.setItem('settings', JSON.stringify(settings));

    // Persist default model choice on the backend too so it survives
    // localStorage clearing or another browser.
    fetch(`${window._env_?.SERVER_URL}/setting/defaults`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ MAIN_MODEL: settings.MAIN_MODEL, SUB_MODEL: settings.SUB_MODEL })
    }).catch(() => {/* best-effort */});

    showToast('设置已保存', 'success');
}

export function showSettings(_previousMode) {
    previousMode = _previousMode;

    // Always rebuild so newly-saved API rows render correctly.
    const existing = document.querySelector('.settings-overlay');
    if (existing) existing.remove();

    const overlay = createSettingsPopup();
    updateModelSelects();
    loadCurrentSettings();
    overlay.style.display = 'block';
}

function hideSettings() {
    const overlay = document.querySelector('.settings-overlay');
    if (overlay) {
        overlay.style.display = 'none';

        if (previousMode) {
            const previousTab = document.querySelector(`.mode-tab[data-value="${previousMode}"]`);
            if (previousTab) {
                previousTab.click();
            }
        }
    }
}
