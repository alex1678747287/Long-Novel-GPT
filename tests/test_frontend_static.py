import re
import unittest
from pathlib import Path


class FrontendWorkbenchStaticTest(unittest.TestCase):
    def test_header_uses_customer_facing_refinement_branding(self):
        source = Path("frontend2/src/App.vue").read_text(encoding="utf-8")
        html = Path("frontend2/index.html").read_text(encoding="utf-8")

        self.assertIn("精修工作台", source)
        self.assertIn("短剧精修引擎", source)
        self.assertIn('to="/settings"', source)
        self.assertIn("模型配置", source)
        self.assertIn('to="/"', source)
        self.assertIn("工作台", source)
        self.assertIn("<title>精修工作台</title>", html)
        self.assertNotIn("洗稿工作台", source)
        self.assertNotIn("<title>洗稿工作台</title>", html)
        self.assertNotIn("短剧改稿引擎", source)

    def test_workbench_hides_script_conversion_from_customer_ui(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for phrase in ["转剧本", "剧本结果", "导出剧本", "复制剧本", "尚未转剧本"]:
            self.assertNotIn(phrase, source)

    def test_quick_add_activates_saved_model(self):
        source = Path("frontend2/src/views/Settings.vue").read_text(encoding="utf-8")

        self.assertIn("const saved = await modelStore.save", source)
        self.assertIn("await modelStore.activate(saved.id)", source)
        self.assertIn("function defaultMaxTokensForPreset", source)
        self.assertIn("max_tokens: defaultMaxTokensForPreset(preset)", source)

    def test_workbench_presents_simple_five_step_flow(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertNotIn('class="workflow-steps"', source)
        for label in ["导入小说", "整本洗稿", "导出文稿"]:
            self.assertIn(label, source)
        for label in ["整本转剧本", "重新转剧本", "转剧本", "导出剧本"]:
            self.assertNotIn(label, source)

    def test_import_passes_chapter_size_to_splitter(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        store = Path("frontend2/src/stores/novel.js").read_text(encoding="utf-8")
        system = Path("frontend2/src/stores/system.js").read_text(encoding="utf-8")

        self.assertIn("max_chapter_size: maxChapterSize.value", workbench)
        self.assertIn("max_chapter_size: meta.max_chapter_size", store)
        self.assertIn("max_chapter_size: options.max_chapter_size", store)
        self.assertIn("max_chapter_size: 2200", system)
        self.assertIn("DEFAULT_MAX_CHAPTER_SIZE = 2200", workbench)

    def test_workbench_uses_model_context_split_targets(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("ONE_MILLION_CONTEXT_MAX_CHAPTER_SIZE = 3000", source)
        self.assertIn("MID_CONTEXT_MAX_CHAPTER_SIZE = 2200", source)
        self.assertIn("function isLargeContextModel", source)
        self.assertIn("function isMidContextModel", source)
        self.assertIn("256000", source)
        self.assertIn("deepseek-v4", source)
        self.assertIn("gpt-5.5", source)
        self.assertIn("modelStore.activeModel", source)
        self.assertIn("ONE_MILLION_CONTEXT_MAX_CHAPTER_SIZE", source)
        self.assertIn("MID_CONTEXT_MAX_CHAPTER_SIZE", source)
        self.assertIn("Math.min(configured, DEFAULT_MAX_CHAPTER_SIZE)", source)
        self.assertNotIn("function isDenseShortParagraphText", source)
        self.assertNotIn("function chapterSizeForText", source)

    def test_workbench_keeps_long_chapters_for_backend_internal_segmentation(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("async function autoSplitOversizedChapters", workbench)
        self.assertIn("await novelStore.resplitActiveNovel({", workbench)
        self.assertIn("api.enqueueNovelRewrite", workbench)
        self.assertNotIn("if (!(await autoSplitOversizedChapters('整本洗稿'))) return", workbench)
        self.assertNotIn("请先重新拆章", workbench)
        self.assertIn("(c.content || '').length > maxChapterSize.value", workbench)
        self.assertIn("检测到", workbench)
        self.assertIn("正在自动拆分", workbench)

    def test_result_pane_hides_variant_tabs_for_simpler_customer_flow(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertNotIn('class="variant-tabs"', workbench)
        self.assertNotIn('class="variant-tab"', workbench)
        self.assertNotIn('@click="viewVariant = \'base\'"', workbench)
        self.assertNotIn('@click="viewVariant = \'script\'"', workbench)
        self.assertIn("导出洗稿", workbench)
        self.assertNotIn("导出剧本", workbench)
        self.assertNotIn("field = 'rewritten'; suffix = '洗稿'\n    return", workbench)
        self.assertNotIn("viewVariant.value = variant\n  const out", workbench)
        self.assertNotIn("剧本结果", workbench)
        self.assertNotIn("尚未转剧本", workbench)
        self.assertNotIn("复制剧本", workbench)

    def test_import_dialog_accepts_zip_as_internal_eval_corpus(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        api = Path("frontend2/src/api/v2.js").read_text(encoding="utf-8")

        self.assertIn("isZip", workbench)
        self.assertIn("parseZipCorpus", api)
        self.assertIn('accept=".txt,.docx,.zip"', workbench)
        self.assertIn("测试资料包", workbench)

    def test_stream_rewrite_errors_when_response_closes_without_done_event(self):
        api = Path("frontend2/src/api/v2.js").read_text(encoding="utf-8")

        self.assertIn("let sawDone = false", api)
        self.assertIn("流式响应提前结束", api)
        self.assertIn("!controller.signal.aborted", api)

    def test_stream_rewrite_has_idle_watchdog(self):
        api = Path("frontend2/src/api/v2.js").read_text(encoding="utf-8")

        self.assertIn("const IDLE_TIMEOUT_MS = 300000", api)
        self.assertIn("resetIdleTimer", api)
        self.assertIn("模型长时间没有返回内容", api)
        self.assertIn("clearIdleTimer", api)

    def test_batch_polling_refreshes_novel_list_after_jobs_finish(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        store = Path("frontend2/src/stores/novel.js").read_text(encoding="utf-8")

        self.assertIn("refreshNovelList", store)
        self.assertIn("activeJobsFinished", workbench)
        self.assertIn("await novelStore.openNovel(novelStore.activeNovelId)", workbench)
        self.assertIn("await novelStore.refreshNovelList()", workbench)

    def test_workbench_blocks_overlapping_generation_jobs(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("function ensureIdle({ requireSlot = false } = {})", workbench)
        self.assertIn("if (!ensureIdle()) return", workbench)
        self.assertIn("if (!ensureIdle({ requireSlot: true })) return", workbench)
        self.assertIn(':disabled="currentRewriteStartDisabled || !novelStore.chapters.length"', workbench)
        self.assertIn(':disabled="currentRewriteStartDisabled || !activeChapter || !hasUsableModel"', workbench)

    def test_stop_rewrite_restores_original_running_chapter(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("const runningJob = ref(null)", workbench)
        self.assertIn("runningJob.value = {", workbench)
        self.assertIn("novelStore.chapters.find((c) => c.id === job.chapterId)", workbench)
        self.assertIn("singleBeforeRun.value?.id === chapter.id", workbench)

    def test_import_keeps_draft_until_create_succeeds(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("const importSubmitting = ref(false)", workbench)
        self.assertIn("function resetImportForm()", workbench)
        self.assertIn("importSubmitting.value = true", workbench)
        self.assertIn("resetImportForm()", workbench)
        self.assertIn(':loading="importSubmitting"', workbench)

    def test_workbench_requires_real_active_model_record(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        models = Path("frontend2/src/stores/models.js").read_text(encoding="utf-8")

        self.assertIn("hasUsableModel", workbench)
        self.assertIn("modelStore.activeModel", workbench)
        self.assertIn("this.activeId = this.models.some((m) => m.id === modelData.active_id)", models)

    def test_workbench_keeps_internal_quality_but_hides_customer_metrics(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        store = Path("frontend2/src/stores/novel.js").read_text(encoding="utf-8")

        self.assertIn("quality_score", store)
        self.assertIn("quality?.delivery_status || quality?.delivery_label || quality?.grade", store)
        self.assertNotIn("qualityScorePct", workbench)
        self.assertNotIn("qualityReviewSummary", workbench)
        self.assertNotIn("quality-review-banner", workbench)
        self.assertNotIn("重合 {{ overlapPct }}%", workbench)
        self.assertNotIn("质量 {{ qualityScorePct }}", workbench)
        self.assertNotIn("重合 ${(ratio * 100).toFixed(1)}%", workbench)
        self.assertNotIn("质量 ${chunk.quality.score}", workbench)
        self.assertNotIn("total_tokens", workbench)

    def test_workbench_uses_auto_quality_mode_for_background_jobs(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        api = Path("frontend2/src/api/v2.js").read_text(encoding="utf-8")

        self.assertIn("DEFAULT_REWRITE_QUALITY_MODE = 'auto'", workbench)
        self.assertIn("enqueueNovelRewrite", api)
        self.assertIn("enqueueChapterRewrite", api)
        self.assertIn("listNovelRewriteJobs", api)
        self.assertIn("cancelRewriteJob", api)
        self.assertIn("pollRewriteJobs", workbench)
        self.assertIn("restoreRewriteJobs", workbench)
        self.assertIn("await api.enqueueNovelRewrite", workbench)
        self.assertIn("await api.enqueueChapterRewrite", workbench)
        self.assertIn("await api.cancelRewriteJob", workbench)
        self.assertNotIn("quality_mode: 'balanced'", workbench)
        self.assertNotIn("runQueueStream(chaptersToRun", workbench)

    def test_overlap_metric_uses_delivery_thresholds(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        api = Path("frontend2/src/api/v2.js").read_text(encoding="utf-8")

        self.assertIn("OVERLAP_EXCELLENT = 0.15", workbench)
        self.assertIn("OVERLAP_DELIVERABLE = 0.22", workbench)
        self.assertIn("v <= OVERLAP_EXCELLENT", workbench)
        self.assertIn("qualityNeedsReview", workbench)
        self.assertNotIn("≤20% 合格", workbench)
        self.assertIn("Math.min(aGrams.size, bGrams.size)", api)

    def test_import_advanced_rewrite_fields_are_hidden_from_customer(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertNotIn("showImportAdvanced", source)
        self.assertNotIn("高级改写方向", source)
        for label in ["题材类目", "目标题材", "文风节奏", "改写强度"]:
            self.assertNotIn(f'label="{label}"', source)

    def test_batch_rewrite_uses_background_job_polling(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("const REWRITE_JOB_POLL_MS", source)
        self.assertIn("activeBatchId", source)
        self.assertIn("activeJobIds", source)
        self.assertIn("function applyRewriteJobSnapshot", source)
        self.assertIn("async function startRewriteJobPolling", source)
        self.assertIn("async function pollRewriteJobs", source)
        self.assertIn("async function stopQueue", source)
        self.assertNotIn("async function runConcurrentQueue", source)
        self.assertNotIn("runConcurrentQueue(chaptersToRun", source)
        self.assertNotIn("runConcurrentQueue(failed", source)

    def test_status_labels_cover_background_rewrite_phases(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for label in ["排队中", "生成中", "质量复查", "自动重试中", "已完成", "待完善", "失败", "已取消"]:
            self.assertIn(label, source)
        self.assertNotIn("备用模型", source)

    def test_workbench_shows_generation_activity_animation_and_eta(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for token in [
            "estimateRewriteSeconds",
            "formatEta",
            "rewriteActivityText",
            "queueEtaText",
            "activity-spinner",
            "mini-pulse",
            "预计约",
            "预计 1 分钟内",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("模型正在思考", source)

    def test_background_rewrite_state_is_scoped_to_origin_novel(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for token in [
            "activeRewriteNovelId",
            "activeRewriteNovelTitle",
            "ACTIVE_REWRITE_CONTEXT_KEY",
            "readPersistedRewriteContext",
            "persistActiveRewriteContext",
            "clearPersistedRewriteContext",
            "novelTitleForRewriteContext",
            "isCurrentNovelStreaming",
            "isOtherNovelStreaming",
            "currentNovelQueueRunning",
            "backgroundRewriteText",
            "background-rewrite-status",
            ":title=\"backgroundRewriteText(context)\"",
            "const pollNovelId = activeRewriteNovelId.value || novelStore.activeNovelId",
            "await pollRewriteContext(pollNovelId)",
            "pollRewriteContext(novelId)",
            "api.listNovelRewriteJobs(\n      novelId",
            "applyRewriteJobSnapshot(jobs, novelId)",
            "targetNovelId !== novelStore.activeNovelId",
            "v-for=\"context in otherRewriteContexts\"",
            ":class=\"{ 'is-running': isCurrentNovelStreaming }\"",
            "if (isOtherNovelStreaming.value) return",
            "isCurrentNovelStreaming.value && running.value",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("v-if=\"queueRunning\"", source)
        self.assertNotIn(":class=\"{ 'is-running': isStreaming }\"", source)

    def test_workbench_supports_multiple_parallel_novel_rewrite_contexts(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for token in [
            "const rewriteContexts = ref({})",
            "allRewriteContexts",
            "currentRewriteContext",
            "otherRewriteContexts",
            "contextProgress(",
            "contextActivityText(",
            "upsertRewriteContext(",
            "removeRewriteContext(",
            "pollRewriteContext(",
            "Object.values(rewriteContexts.value)",
            "if (isCurrentNovelStreaming.value)",
            "v-for=\"context in otherRewriteContexts\"",
            "MAX_PARALLEL_REWRITE_NOVELS = 3",
            "atParallelNovelLimit",
            "currentRewriteStartDisabled",
            "最多同时洗 ${MAX_PARALLEL_REWRITE_NOVELS} 本小说",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("if (!isStreaming.value) return true\n  const title = activeRewriteNovelTitle.value", source)

    def test_batch_jobs_do_not_depend_on_long_frontend_streams(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("api.enqueueNovelRewrite", source)
        self.assertIn("api.listNovelRewriteJobs", source)
        self.assertIn("REWRITE_JOB_POLL_MS", source)
        self.assertIn("cancelActiveRewriteJobs", source)
        self.assertNotIn("function isRetryableStreamError", source)
        self.assertNotIn("async function runQueueStream", source)

    def test_rewrite_jobs_apply_saved_results_from_polling(self):
        workbench = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")
        store = Path("frontend2/src/stores/novel.js").read_text(encoding="utf-8")

        self.assertIn("function applyRewriteJobSnapshot", workbench)
        self.assertIn("job.result?.quality?.issues", workbench)
        self.assertIn("chapter.rewritten = text", workbench)
        self.assertIn("chapter.quality_issues = JSON.stringify", workbench)
        self.assertIn("await novelStore.refreshNovelList()", workbench)
        self.assertIn("return api", store)
        self.assertIn("return true", store)
        self.assertIn("return false", store)

    def test_workbench_uses_fixed_export_document_menu(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("handleExportCommand", source)
        self.assertIn("导出文稿", source)
        self.assertIn('command="source"', source)
        self.assertIn('command="base"', source)
        self.assertNotIn('command="script"', source)

    def test_workbench_blocks_incomplete_generated_exports(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("function generatedExportMissingCount", source)
        self.assertIn("async function generatedExportReviewCount", source)
        self.assertIn("api.qualityScore", source)
        self.assertIn("待完善", source)
        self.assertIn("confirmButtonText: '继续导出'", source)
        self.assertIn("洗稿还有", source)
        self.assertNotIn("剧本还有", source)
        self.assertNotIn("[未生成]", source)

    def test_workbench_marks_reviewable_done_chapters_as_polishable(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        for token in [
            "function chapterNeedsPolish",
            "return '待完善'",
            "jobQualityIssues(job).length ? '待完善'",
            "正在整理人物/世界观/情节线",
            "ensureRewriteAnalysisReady",
            "await api.reanalyzeNovel",
            "人物/世界观/情节线整理失败",
            "仍有 ${reviewCount} 章待完善",
            "返回检查",
        ]:
            self.assertIn(token, source)

    def test_prompt_selection_is_not_on_main_workbench(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertNotIn("<label>Prompt</label>", source)
        self.assertNotIn("<label>剧本模板</label>", source)

    def test_workbench_hides_internal_analysis_status_wording(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertNotIn("人物已统一对照", source)
        self.assertNotIn("分析失败", source)

    def test_workbench_uses_customer_facing_prompt_error_messages(self):
        source = Path("frontend2/src/views/Workbench.vue").read_text(encoding="utf-8")

        self.assertIn("改写能力未加载", source)
        self.assertNotIn("洗稿模板", source)
        self.assertNotIn("prompt 模板", source)
        self.assertNotIn("后端 prompts 配置", source)

    def test_header_model_dropdown_hides_model_id_details(self):
        source = Path("frontend2/src/App.vue").read_text(encoding="utf-8")

        self.assertNotIn("model-opt-id", source)
        self.assertNotIn("{{ m.model }}", source)

    def test_settings_page_hides_prompt_and_system_tabs(self):
        source = Path("frontend2/src/views/Settings.vue").read_text(encoding="utf-8")

        self.assertNotIn('label="Prompt 模板"', source)
        self.assertNotIn('label="系统参数"', source)
        self.assertNotIn("自定义 Prompt 模板", source)
        self.assertNotIn("系统参数已保存", source)

    def test_settings_quick_add_is_api_key_first(self):
        source = Path("frontend2/src/views/Settings.vue").read_text(encoding="utf-8")

        self.assertIn("showQuickAddAdvanced", source)
        self.assertIn("更多模型设置", source)
        quick_add_match = re.search(
            r'<div v-if="quickAddPreset" class="quick-add-body">.*?</div>\s*<template #footer>',
            source,
            re.S,
        )
        self.assertIsNotNone(quick_add_match)
        visible_before_advanced = quick_add_match.group(0).split('v-if="showQuickAddAdvanced"', 1)[0]
        self.assertIn("API Key", visible_before_advanced)
        self.assertNotIn("显示名称", visible_before_advanced)
        self.assertNotIn("模型 ID", visible_before_advanced)

    def test_settings_model_cards_hide_endpoint_and_key_details(self):
        source = Path("frontend2/src/views/Settings.vue").read_text(encoding="utf-8")

        configured_section = source.split("<!-- Configured models -->", 1)[1].split("<!-- Quick-add dialog", 1)[0]
        self.assertNotIn("端点", configured_section)
        self.assertNotIn("Key", configured_section)
        self.assertNotIn("高级 / 自定义", configured_section)
        self.assertNotIn("{{ m.model }}", configured_section)
        self.assertNotIn("model-chip", configured_section)

    def test_settings_provider_cards_hide_default_model_ids(self):
        source = Path("frontend2/src/views/Settings.vue").read_text(encoding="utf-8")

        preset_section = source.split("<!-- Preset cards", 1)[1].split("<!-- Configured models", 1)[0]
        self.assertIn("粘贴密钥即可使用", preset_section)
        self.assertNotIn("p.default_model", preset_section)
