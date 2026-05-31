from core.writer_utils import KeyPointMsg
from core.writer import Writer

from prompts.提炼.prompt import main as prompt_summary

LEGACY_PROMPTS_REMOVED = "旧版创作提示词目录已移除，请使用新版 /v2/rewrite 或 /v2/prompts。"

class PlotWriter(Writer):
    def __init__(self, xy_pairs, global_context, model=None, sub_model=None, x_chunk_length=200, y_chunk_length=1000, max_thread_num=5):
        super().__init__(xy_pairs, global_context, model, sub_model, x_chunk_length=x_chunk_length, y_chunk_length=y_chunk_length, max_thread_num=max_thread_num)

    def write(self, user_prompt, pair_span=None):
        raise RuntimeError(LEGACY_PROMPTS_REMOVED)
    
    def summary(self):
        target_chunk = self.get_chunk(pair_span=(0, len(self.xy_pairs)))
        if not target_chunk.y_chunk:
            raise Exception("没有剧情需要总结。")
        if len(target_chunk.y_chunk) <= 5:
            raise Exception("需要总结的剧情不能少于5个字。")
        
        result = yield from prompt_summary(self.model, "提炼章节", y=target_chunk.y_chunk)

        self.global_context['chapter'] = result['text']

    def get_model(self):
        return self.model

    def get_sub_model(self):
        return self.sub_model
