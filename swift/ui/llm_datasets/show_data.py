from typing import Type

import gradio as gr

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelType
from swift.ui.base import BaseUI
import os


base_path = '/openbayes/home/swift/openbayes-beryllium/data'
absolute_path = None
complex_files = []
global output_base
## 判断数字函数
def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False
## 根据文件夹名称中的字段(如sqa)选择希望展示在下拉框中的数据集，第一行中的六个按钮皆调用此函数
def update_list_files(name):
    try:
        datasetdir = [""]
        for root, dirs, _ in os.walk(base_path):
            for dir in dirs:
                if name in dir:  # 检查文件夹名称是否包含 'name'
                    full_path = os.path.join(root, dir)
                    datasetdir.append(full_path)
            break  # 只需要遍历一级子文件夹，因此可以跳出当前循环
        datasetdir.sort()
        display_choices = [os.path.basename(path) for path in datasetdir]
        return (gr.update(choices=display_choices, value=display_choices[0]),gr.update(visible=False),
                gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),
                gr.update(value=1),gr.update(value=100),gr.update(value=1),gr.update(value=100),gr.update(value=""))
    except Exception as e:
        return (gr.update(choices=display_choices, value=display_choices[0]),gr.update(visible=False),
                gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),
                gr.update(value=1),gr.update(value=100),gr.update(value=1),gr.update(value=100),gr.update(value=""))

## 数据内容读取函数，根据文件后缀名选择对应的展示方式
def load_file(file_path, start_line, end_line):
    if os.path.exists(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()

        ## 图片文件展示，列表中为目前可供查看的图片文件类型（待补全）
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.mng', '.eps']:
            return file_path, None, None, None, None, "image"
        
        ## 视频文件展示，列表中为目前可供查看的视频文件类型（待补全，视频文件最好是“.mp4”格式，如果是其他视频格式则会在目标文件路径下创建一个“.mp4”格式的文件）
        elif file_ext in ['.mp4', '.avi']:
            return None, file_path, None, None, None, "video"
        
        ## PDF文件展示
        elif file_ext in ['.pdf']:
            return None, None, file_path, None, None, "pdf"
        
        ## 文本文件展示，列表中为目前可供查看的文本文件类型（待补全,只要文件内容属于文本，都可以将其后缀添加进来）,文本文件可选择查看具体数据中的某几行
        elif file_ext in ['.txt', '.sh', '.log', '.py', '.toml', '.doc', '.docx', '.json', '.jsonl', '.csv']:

            lines_to_display = []

            # 计算文件行数
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f)
                
            if start_line and end_line:
                try:
                    start_line, end_line = int(start_line), int(end_line)
                    # 确保行数在有效范围内
                    start_line = max(1, start_line)
                    end_line = min(line_count, end_line)
                    if start_line > end_line:
                        text_content = "Invalid line range: start line should be less than or equal to end line."
                    else:
                        with open(os.path.join(file_path), 'r', encoding='utf-8') as file:
                            for current_line_number, line in enumerate(file, start=1):
                                if start_line <= current_line_number <= end_line:
                                    lines_to_display.append(f'『第{current_line_number}行』:   {line} \n')
                        text_content = "".join(lines_to_display)
                except ValueError:
                    text_content = "Invalid line numbers entered."
            else:
                text_content = "Please enter valid line numbers."
                
            return None, None, None, text_content, line_count, "text"
        
        elif os.path.isdir(file_path):
            return None, None, None, "当前查询路径为文件夹。(文件夹中的文件请等待 “多模态源文件列表” 加载后进行选择，若文件夹过大（如 llava 文件夹内含 56 万张图片，初始化加载大约需要28秒）则可能需要一部分的等待时间......)", None, "text"

        ## 未知文件类型
        else:
            return None, None, None, "该文件类型暂未添加到脚本中，请手动添加，脚本路径：/openbayes-beryllium/gradioapp.py", None, "text"
    else:
        return None, None, None, f"文件 {file_path} 不存在，请检查后重试！", None, "text"


# data_type_list = ['DPO数据集','MI数据集','SQA数据集','MQA数据集','TXT数据集','CODE数据集']
types_dict = {
    'DPO数据集':'dpo',
    'MI数据集':'mi',
    'SQA数据集':'sqa',
    'MQA数据集':'mqa',
    'TXT数据集':'txt',
    'CODE数据集':'code'

}


class DATA_SHOW(BaseUI):
    group = 'llm_datasets'

    locale_dict = {
        'model_type': {
            'label': {
                'zh': '选择数据集类型',
                'en': 'Select Model'
            },
            'info': {
                'zh': 'openbayes  已支持的数据集类型',
                'en': 'Base model supported by SWIFT'
            }
        },
        'template_type': {
            'label': {
                'zh': '选择数据集',
                'en': 'Prompt template type'
            },
            'info': {
                'zh': '选择实际使用的数据集',
                'en': 'Choose the template type of the model'
            }
        },
        'clear_cache': {
            'value': {
                'zh': '展示数据',
                'en': 'Delete train records'
            },
        },
        'show_readme': {
            'value': {
                'zh': '数据集介绍',
                'en': 'Delete train records'
            },
        },
        'sel_page': {
            'value': {
                'zh': '跳到该页',
                'en': 'Delete train records'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):

        global output_base
        with gr.Accordion(elem_id='model_param', open=True):
        # with gr.TabItem(elem_id='llm_train', label=''):
            with gr.Row():
                dataset_type = gr.Dropdown(
                    elem_id='model_type',
                    choices=types_dict.keys(),
                    scale=20)
                # model_id_or_path = gr.Textbox(elem_id='model_id_or_path', lines=1, scale=20, interactive=True)

                template_type = gr.Dropdown(
                    elem_id='template_type', choices=types_dict.keys(), scale=20)
                # train_record = gr.Dropdown(elem_id='train_record', choices=[], scale=20)


                with gr.Column():
                    clear_cache = gr.Button(elem_id='clear_cache', scale=2, variant='primary')
                    show_readme = gr.Button(elem_id='show_readme', scale=2, variant='primary')
                model_state = gr.State({})


            with gr.Row():
                with gr.Column(visible=False) as show_files:
                    with gr.Row():
                        origin_list = gr.Dropdown(
                            elem_id='origin',
                            choices=types_dict.keys(),
                            scale=20)
                        output_list = gr.Dropdown(
                            elem_id='output', choices=types_dict.keys(), scale=20)

        # 展示origin 和 output
        with gr.Accordion(elem_id='showfile', open=True):

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        page_num = gr.Textbox(label="请输入显示第几条数据", value="1")
                        page_num_out = gr.Textbox(label="请输入显示第几条数据", value="1")
                        
                    sel_page = gr.Button(elem_id='sel_page', scale=1, variant='primary')
                    with gr.Row():
                        control_origin = gr.JSON()
                        control_output = gr.JSON()
                    # control_output = gr.Row()


            
        # 展示readme
        with gr.Accordion(elem_id='readm', open=True):
            readme_output = gr.Markdown()

        # =======================================================================================================================
        # ====================================操作函数============================================================================
        # =======================================================================================================================

        def show_json(filename,page):
            import json
            # JSON 文件路径
            
            result = []
            # 使用 with 语句打开文件，这样可以确保文件在读取后会被正确关闭
            with open(filename, 'r', encoding='utf-8') as file:
                # 加载 JSON 数据
                for line in file.readlines():
                    data = json.loads(line)
                    result.append(data)
            # 现在 data 变量包含了 JSON 文件中的数据，可以作为 Python 的字典或列表使用
            if int(page) < 0:
                page = 0
            elif int(page) >= len(result):
                page = len(result) -1
            else :
                pass
            return result[int(page)]

        def choose_origin_model(filename1,filename2,page):
            
            if 'json' in filename2:
                print(' json ')
                file_path = os.path.join(base_path,filename1,'origin',filename2)
                json_list = show_json(file_path,page)
                # json_component = gr.JSON(value={"key1": "value1","key2": 123,"key3": [1, 2, 3],"key4": {"nestedKey1": "nestedValue1"}})
                return json_list


            if 'csv' in filename2:
                print(' csv ')
                file_path = os.path.join(base_path,filename1,'origin',filename2)
                json_list = show_json(file_path)
                # json_component = gr.JSON(value={"key1": "value1","key2": 123,"key3": [1, 2, 3],"key4": {"nestedKey1": "nestedValue1"}})
                return json_list

            else:
                print(' nono ')
                out_text_output = gr.Textbox(label="暂不支持此类型文件")
                return []
            

        def choose_output_model(filename1,filename2,page):
            
            if 'json' in filename2:
                print(' json ')
                file_path = os.path.join(base_path,filename1,output_base,filename2)
                json_list = show_json(file_path,page)
                # json_component = gr.JSON(value={"key1": "value1","key2": 123,"key3": [1, 2, 3],"key4": {"nestedKey1": "nestedValue1"}})
                return json_list


            if 'csv' in filename2:
                print(' csv ')
                file_path = os.path.join(base_path,filename1,output_base,filename2)
                json_list = show_json(file_path)
                # json_component = gr.JSON(value={"key1": "value1","key2": 123,"key3": [1, 2, 3],"key4": {"nestedKey1": "nestedValue1"}})
                return json_list
            else:
                print(' nono ')
                out_text_output = gr.Textbox(label="暂不支持此类型文件")
                return []
        

        def sel_all_pages(filename1,origin_file,out_file,page1,page2):
            print(page1)
            print(page2)
            json_origin = []
            json_out = []
            if 'json' in origin_file:
                print(' json ')
                file_path = os.path.join(base_path,filename1,'origin',origin_file)
                json_origin = show_json(file_path,page1)

            if 'json' in out_file:
                file_path_out = os.path.join(base_path,filename1,output_base,out_file)
                json_out = show_json(file_path_out,page2)
                # json_component = gr.JSON(value={"key1": "value1","key2": 123,"key3": [1, 2, 3],"key4": {"nestedKey1": "nestedValue1"}})
                
            if 'csv' in origin_file:
                pass
            if 'csv' in out_file:
                pass
            
            return json_origin,json_out


        def show_origin(base_name):
            origin_path = os.path.join(base_path,base_name,'origin')
            if os.path.exists(origin_path):
                origin_files = os.listdir(origin_path)
                return gr.update(choices=origin_files, value=origin_files[0])
            else :
                print('这个数据集是坏的')
                return gr.update(choices=[], value='None') 

        def show_output(base_name):
            global output_base
            abs_path = os.path.join(base_path,base_name)
            outputs = os.listdir(abs_path)
            for basename in outputs:
                if 'output' in basename :
                    output_base = basename
                    output_path = os.path.join(abs_path,basename)
                    if os.path.isdir(output_path):
                        output_files = os.listdir(output_path)
                        return gr.update(choices=output_files, value=output_files[0])
            return gr.update(choices=[], value='None') 

        def show_results(base_name):
            
            return show_origin(base_name) ,show_output(base_name) ,gr.update(visible=True)

        def list_dataset(datas_types):
            datasets = os.listdir(base_path)
            sel_dataset = []
            for dataset in datasets:
                if datas_types in dataset:
                    sel_dataset.append(dataset)
            return sel_dataset

        def display_folder(file_path):
            global absolute_path
            # absolute_path = os.path.join(absolute_path,file_path)
            datas_types = types_dict[file_path]
            dataset_lists = list_dataset(datas_types)

            # return dataset_lists
            if len(dataset_lists) > 0:
                return gr.update(choices=dataset_lists, value=dataset_lists[0]) 
            else:
                return gr.update(choices=[], value='None') 

        def display_md_file(file_path):

            md_path = os.path.join(base_path,file_path,file_path+'.md')

            if os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                return gr.Markdown(content)
            else:
                return gr.Markdown("文件不存在")

        # ==============================s21ep837m51==========================================================

        dataset_type.change(fn=display_folder, inputs=dataset_type, outputs=[template_type])
        origin_list.change(fn=choose_origin_model,inputs=[template_type,origin_list,page_num],outputs=control_origin)
        output_list.change(fn=choose_output_model,inputs=[template_type,output_list,page_num_out],outputs=control_output)
        clear_cache.click(fn=show_results, inputs=template_type, outputs=[origin_list,output_list,show_files])
        show_readme.click(fn=display_md_file, inputs=template_type, outputs=readme_output)
        sel_page.click(fn=sel_all_pages,inputs=[template_type,origin_list,output_list,page_num,page_num_out],outputs=[control_origin,control_output])



