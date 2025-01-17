import os
from typing import Type

import gradio as gr

from swift.llm import DATASET_MAPPING
from swift.ui.base import BaseUI
from swift.ui.llm_train.CONFIG import OPENBAYES_DATASET_MAPPING

class Dataset(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'dataset': {
            'label': {
                'zh': '数据集名称',
                'en': 'Dataset Code'
            },
            'info': {
                'zh': '选择训练的数据集，支持复选',
                'en': 'The dataset(s) to train the models'
            }
        },
        'openbayes_dataset': {
            'label': {
                'zh': 'openbayes 数据集名称',
                'en': 'Dataset Code'
            },
            'info': {
                'zh': '选择训练的数据集，支持复选',
                'en': 'The dataset(s) to train the models'
            }
        },
        'max_length': {
            'label': {
                'zh': '句子最大长度',
                'en': 'The max length',
            },
            'info': {
                'zh': '设置输入模型的最大长度',
                'en': 'Set the max length input to the model',
            }
        },
        'dataset_test_ratio': {
            'label': {
                'zh': '验证集拆分比例',
                'en': 'Split ratio of eval dataset'
            },
            'info': {
                'zh': '表示将总数据的多少拆分到验证集中',
                'en': 'Split the datasets by this ratio for eval'
            }
        },
        'train_dataset_sample': {
            'label': {
                'zh': '训练集采样数量',
                'en': 'The sample size from the train dataset'
            },
            'info': {
                'zh': '从训练集中采样一定行数进行训练',
                'en': 'Train with the sample size from the dataset',
            }
        },
        'val_dataset_sample': {
            'label': {
                'zh': '验证集采样数量',
                'en': 'The sample size from the val dataset'
            },
            'info': {
                'zh': '从验证集中采样一定行数进行训练',
                'en': 'Validate with the sample size from the dataset',
            }
        },
        'custom_dataset_info': {
            'label': {
                'zh': '外部数据集配置',
                'en': 'Custom dataset config'
            },
            'info': {
                'zh': '注册外部数据集的配置文件',
                'en': 'An extra dataset config to register your own datasets'
            }
        },
        'openbayes_dataset_info': {
            'label': {
                'zh': 'openbayes数据集选择',
                'en': 'Custom dataset config'
            },
            'info': {
                'zh': 'openbayes数据集选择',
                'en': 'An extra dataset config to register your own datasets'
            }
        },
        'dataset_param': {
            'label': {
                'zh': '数据集设置',
                'en': 'Dataset settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='dataset_param', open=True):
            with gr.Row():
                gr.Dropdown(
                    elem_id='dataset',
                    multiselect=True,
                    choices=list(DATASET_MAPPING.keys()),
                    scale=20,
                    allow_custom_value=True)
                gr.Textbox(elem_id='custom_dataset_info', is_list=False, scale=20)

            with gr.Row():
                with gr.Column():
                    gr.Dropdown(
                        elem_id='openbayes_dataset',
                        multiselect=True,
                        choices=list(OPENBAYES_DATASET_MAPPING),
                        scale=20,
                        allow_custom_value=True)
            with gr.Row():
                gr.Slider(elem_id='dataset_test_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=20)
                gr.Slider(elem_id='max_length', minimum=32, maximum=32768, step=1, scale=20)
                gr.Textbox(elem_id='train_dataset_sample', scale=20)
                gr.Textbox(elem_id='val_dataset_sample', scale=20)
                