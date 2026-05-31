import copy
import functools
import pickle
import os
import time
import gradio as gr

from core.writer import Chunk

title = """
<div style="text-align: center; padding: 10px 20px;">
    <h1 style="margin: 0 0 5px 0;">🖋️ Long-Novel-GPT 1.10</h1>
    <p style="margin: 0;"><em>AI一键生成长篇小说</em></p>
</div>
"""

info = \
"""1. 当前Demo支持GPT、Claude、文心、豆包、GLM等模型，并且已经配置了API-Key，默认模型为GPT4o，最大线程数为5。
2. 可以选中**示例**中的任意一个创意，然后点击**创作大纲**来初始化大纲。
3. 初始化后，点击**开始创作**按钮，可以不断创作大纲，直到满意为止。
4. 创建完大纲后，点击**创作剧情**按钮，之后重复以上流程。
5. 选中**一键生成**后，再次点击左侧按钮可以一键生成。
6. 如果遇到任何无法解决的问题，请点击**刷新**按钮。
7. 如果问题还是无法解决，请刷新浏览器页面，这会导致丢失所有数据，请手动备份重要文本。
"""


def init_writer(idea, check_empty=True):  
    outline_w = dict(
        current_cost=0,
        total_cost=0,
        currency_symbol='￥',
        xy_pairs=[(idea, '')],
        apply_chunks={},
    )
    chapters_w = dict(
        current_cost=0,
        total_cost=0,
        currency_symbol='￥',
        xy_pairs=[('', '')],
        apply_chunks={},
    )
    draft_w = dict(
        current_cost=0,
        total_cost=0,
        currency_symbol='￥',
        xy_pairs=[('', '')],
        apply_chunks={},
    )
    suggestions = dict(
        outline_w = ['新建大纲', '扩写大纲', '润色大纲'],
        chapters_w = ['新建剧情', '扩写剧情', '润色剧情'],
        draft_w = ['新建正文', '扩写正文', '润色正文'],
    )

    suggestions_dirname = dict(
        outline_w = None,
        chapters_w = None,
        draft_w = None,
    )

    chunk_length = dict(
        outline_w = [4_000, ],
        chapters_w = [500, 200, 1000, 2000],
        draft_w = [1000, 500, 2000, 3000],
    )

    writer = dict(
        current_w='outline_w',
        outline_w=outline_w,
        chapters_w=chapters_w,
        draft_w=draft_w,
        running_flag=False,
        cancel_flag=False,  # 用于取消正在进行的操作
        pause_flag=False,   # 用于暂停操作
        progress={},
        prompt_outputs=[],  # 这一行未注释时，将在gradio界面中显示prompt_outputs
        suggestions=suggestions,
        suggestions_dirname=suggestions_dirname,
        pause_on_prompt_finished_flag = False,
        quote_span = None,
        chunk_length = chunk_length,
    )

    current_w_name = writer['current_w']
    if check_empty and writer_x_is_empty(writer, current_w_name):
        raise Exception('请先输入小说简介！')
    else:
        return writer

def init_chapters_w(writer, check_empty=True):
    outline_w = writer['outline_w']
    chapters_w = writer['chapters_w']
    outline_y = "".join([e[1] for e in outline_w['xy_pairs']])
    chapters_w['xy_pairs'] = [(outline_y, '')]

    writer["current_w"] = "chapters_w"
    
    current_w_name = writer['current_w']
    if check_empty and writer_x_is_empty(writer, current_w_name):
        raise Exception('大纲不能为空')
    else:
        return writer

def init_draft_w(writer, check_empty=True):
    chapters_w = writer['chapters_w']
    draft_w = writer['draft_w']
    chapters_y = "".join([e[1] for e in chapters_w['xy_pairs']])
    draft_w['xy_pairs'] = [(chapters_y, '')]

    writer["current_w"] = "draft_w"
    
    current_w_name = writer['current_w']
    if check_empty and writer_x_is_empty(writer, current_w_name):
        raise Exception('剧情不能为空')
    else:
        return writer

# 在将writer传递到backend之前，只传递backend需要的部分
# 这样从backend返回new_writer后，可以直接用update更新writer_state
def process_writer_to_backend(writer):
    remained_keys = ['current_w', 'outline_w', 'chapters_w', 'draft_w', 'quote_span']
    new_writer = {key: writer[key] for key in remained_keys}
    return copy.deepcopy(new_writer)

# 在整个writer_state生命周期中，其对象地址都不应被改变，这样方便各种flag的检查
def process_writer_from_backend(writer, new_writer):
    for key in ['outline_w', 'chapters_w', 'draft_w']:
        writer[key] = copy.deepcopy(new_writer[key])
    return writer

def is_running(writer):
    # 只检查是否有正在运行的操作
    return writer['running_flag'] and not writer['cancel_flag']

def has_accept(writer):
    # 只检查是否有待接受的文本
    current_w = writer[writer['current_w']]
    return bool(current_w['apply_chunks'])

def cancellable(func):
    @functools.wraps(func)
    def wrapper(writer, *args, **kwargs):
        if is_running(writer):
            gr.Warning('另一个操作正在进行中，请等待其完成或取消！')
            return
        
        if has_accept(writer) and wrapper.__name__ != "on_accept_write":
            gr.Warning('有正在等待接受的文本，点击接受或取消！')
            return
        
        writer['running_flag'] = True
        writer['cancel_flag'] = False
        writer['pause_flag'] = False
        
        generator = func(writer, *args, **kwargs)
        result = None
        try:
            while True:   
                if writer['cancel_flag']:
                    gr.Info('操作已取消！')
                    return
                
                # pause 暂停逻辑由func内部实现，便于它们在暂停前后执行一些操作              
                try:
                    result = next(generator)
                    if isinstance(result, tuple) and (writer_dict := next((item for item in result if isinstance(item, dict) and 'running_flag' in item), None)):
                        assert writer is writer_dict, 'writer对象地址发生了改变'
                        writer = writer_dict
                    yield result
                except StopIteration as e:
                    return e.value
                except Exception as e:
                    raise gr.Error(f'操作过程中发生错误：{e}')
        finally:
            writer['running_flag'] = False
            writer['pause_flag'] = False
    
    return wrapper

def try_cancel(writer):
    if not (is_running(writer) or has_accept(writer)):
        gr.Info('当前没有正在进行的操作或待接受的文本')
        return
    
    writer['prompt_outputs'] = []
    current_w = writer[writer['current_w']]
    if not is_running(writer) and has_accept(writer):    # 优先取消待接受的文本
        current_w['apply_chunks'].clear()
        gr.Info('已取消待接受的文本')
        return

    writer['cancel_flag'] = True
    
    start_time = time.time()
    while writer['running_flag'] and time.time() - start_time < 3:
        time.sleep(0.1)
    
    if writer['running_flag']:
        gr.Warning('取消操作超时，可能需要刷新页面')
    
    writer['cancel_flag'] = False
    
def writer_y_is_empty(writer, w_name):
    xy_pairs = writer[w_name]['xy_pairs']
    return sum(len(e[1]) for e in xy_pairs) == 0

def writer_x_is_empty(writer, w_name):
    xy_pairs = writer[w_name]['xy_pairs']
    return sum(len(e[0]) for e in xy_pairs) == 0


# create a markdown table
# TODO: 优化显示逻辑，字少的列宽度小，字多的列宽度大
def create_comparison_table(pairs, column_names=['Original Text', 'Enhanced Text', 'Enhanced Text 2']):
    # Check if any pair has 3 elements
    has_third_column = any(len(pair) == 3 for pair in pairs)
    
    # Create table header
    if has_third_column:
        table = f"| {column_names[0]} | {column_names[1]} | {column_names[2]} |\n|---------------|-----------------|----------------|\n"
    else:
        table = f"| {column_names[0]} | {column_names[1]} |\n|---------------|---------------|\n"
    
    # Add rows to the table
    for pair in pairs:
        x = pair[0].replace('|', '\\|').replace('\n', '<br>')
        y1 = pair[1].replace('|', '\\|').replace('\n', '<br>')
        
        if has_third_column:
            y2 = pair[2].replace('|', '\\|').replace('\n', '<br>') if len(pair) == 3 else ''
            table += f"| {x} | {y1} | {y2} |\n"
        else:
            table += f"| {x} | {y1} |\n"
    
    return table

def messages2chatbot(messages):
    if len(messages) and messages[0]['role'] == 'system':
        return [{'role': 'user', 'content': messages[0]['content']}, ] + messages[1:]
    else:
        return messages
    
def create_progress_md(writer):
    progress_md = ""
    if 'progress' in writer and writer['progress']:
        progress = writer['progress']
        progress_md = ""
        
        # 使用集合来去重并保持顺序
        titles = []
        subtitles = {}
        current_op_ij = (float('inf'), float('inf'))
        for opi, op in enumerate(progress['ops']):
            if op['title'] not in titles:
                titles.append(op['title'])
            if op['title'] not in subtitles:
                subtitles[op['title']] = []
            if op['subtitle'] not in subtitles[op['title']]:
                subtitles[op['title']].append(op['subtitle'])
            
            if opi == progress['cur_op_i']:
                current_op_ij = (len(titles), len(subtitles[op['title']]))
        
        for i, title in enumerate(titles, 1):
            progress_md += f"## {['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'][i-1]}、{title}\n"
            for j, subtitle in enumerate(subtitles[title], 1):
                if i < current_op_ij[0] or (i == current_op_ij[0] and j < current_op_ij[1]):
                    progress_md += f"### {j}、{subtitle} ✓\n"
                elif i == current_op_ij[0] and j == current_op_ij[1]:
                    progress_md += f"### {j}、{subtitle} {'.' * (int(time.time()) % 4)}\n"
                else:
                    progress_md += f"### {j}、{subtitle}\n"
            
            progress_md += "\n"
        
        progress_md += "---\n"
        # TODO: 考虑只放当前进度

    return gr.Markdown(progress_md)

                
def create_text_md(writer):
    current_w_name = writer['current_w']
    current_w = writer[current_w_name]
    apply_chunks = current_w['apply_chunks']

    match current_w_name:
        case 'draft_w':
            column_names = ['剧情', '正文', '修正稿']
        case 'outline_w':
            column_names = ['小说简介', '大纲', '修正稿']
        case 'chapters_w':
            column_names = ['大纲', '剧情', '修正稿']
        case _:
            raise Exception('当前状态不正确')

    xy_pairs = current_w['xy_pairs']
    if apply_chunks:
        table = [[*e, ''] for e in xy_pairs]
        occupied_rows = [False] * len(table)
        for chunk, key, text in apply_chunks:
            if not isinstance(chunk, Chunk):
                chunk = Chunk(**chunk)
            assert key == 'y_chunk'
            pair_span = chunk.text_source_slice
            if any(occupied_rows[pair_span]):
                raise Exception('apply_chunks中存在重叠的pair_span')
            occupied_rows[pair_span] = [True] * (pair_span.stop - pair_span.start)
            table[pair_span] = [[chunk.x_chunk, chunk.y_chunk, text], ] + [None] * (pair_span.stop - pair_span.start - 1)
        table = [e for e in table if e is not None]
        if not any(e[1] for e in table):
            column_names = column_names[:2]
            column_names[1] = column_names[1] + '（待接受）'
            table = [[e[0], e[2]] for e in table]
        md = create_comparison_table(table, column_names=column_names)
    else:
        if writer_x_is_empty(writer, current_w_name):
            tip_x = '从下方示例中选择一个创意用于创作小说。'
            tip_y = '选择创意后，点击创作大纲。更详细的操作请参考使用指南。'
            if not xy_pairs[0][0].strip():
                xy_pairs = [[tip_x, tip_y]]
            else:
                xy_pairs = [[xy_pairs[0][0], tip_y]]

        md = create_comparison_table(xy_pairs, column_names=column_names[:2])
    
    if len(md) < 400:
        height = '200px'
    else:
        height = '600px'
    return gr.Markdown(md, height=height)
