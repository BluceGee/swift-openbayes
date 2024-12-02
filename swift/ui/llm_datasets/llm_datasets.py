import os
import re
from typing import Dict, Type
import os
import re
import gradio as gr
from gradio_pdf import PDF
from natsort import natsorted
import gradio as gr
from swift.llm import RLHFArguments
from swift.ui.base import BaseUI
from swift.ui.llm_datasets.show_data import DATA_SHOW
from swift.utils import get_logger
logger = get_logger()


is_spaces = True if 'SPACE_ID' in os.environ else False
if is_spaces:
    is_shared_ui = True if 'modelscope/swift' in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False


class LLDatasets(BaseUI):
    group = 'llm_datasets'

    is_studio = os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio'

    sub_ui = [
        DATA_SHOW
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_datasets': {
            'label': {
                'zh': 'openbayes-beryllium',
                'en': 'openbayes-beryllium',
            }
        },
        'train_type': {
            'label': {
                'zh': '训练Stage',
                'en': 'Train Stage'
            },
            'info': {
                'zh': '请注意选择于此匹配的数据集，人类对齐配置在页面下方',
                'en': 'Please choose matched dataset, RLHF settings is at the bottom of the page'
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程',
                'en':
                'Task started, please check the tensorboard or log file, '
                'closing this page does not affect training'
            }
        },
        'dataset_alert': {
            'value': {
                'zh': '请选择或填入一个数据集',
                'en': 'Please input or select a dataset'
            }
        },
        'submit': {
            'value': {
                'zh': '🚀 开始训练',
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'zh': '仅生成运行命令',
                'en': 'Dry-run'
            },
            'info': {
                'zh': '仅生成运行命令，开发者自行运行',
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
        'sft_type': {
            'label': {
                'zh': '训练方式',
                'en': 'Train type'
            },
            'info': {
                'zh': '选择训练的方式',
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'zh': '随机数种子',
                'en': 'Seed'
            },
            'info': {
                'zh': '选择随机数种子',
                'en': 'Select a random seed'
            }
        },
        'dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'use_ddp': {
            'label': {
                'zh': '使用DDP',
                'en': 'Use DDP'
            },
            'info': {
                'zh': '是否使用数据并行训练',
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'ddp_num': {
            'label': {
                'zh': 'DDP分片数量',
                'en': 'Number of DDP sharding'
            },
            'info': {
                'zh': '启用多少进程的数据并行',
                'en': 'The data parallel size of DDP'
            }
        },
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'tuner实现框架，建议peft或者unsloth',
                'en': 'The tuner backend, suggest to use peft or unsloth'
            }
        },
        'use_liger': {
            'label': {
                'zh': '使用Liger kernel',
                'en': 'Use Liger kernel'
            },
            'info': {
                'zh': 'Liger kernel可以有效降低显存使用',
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'zh': '序列并行分段',
                'en': 'Sequence parallel size'
            },
            'info': {
                'zh': 'DDP条件下的序列并行（减小显存），需要安装ms-swift[seq_parallel]',
                'en': 'Sequence parallel when ddp, need to install ms-swift[seq_parallel]'
            }
        },
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    choice_dict = BaseUI.get_choices_from_dataclass(RLHFArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(RLHFArguments)
    arguments = BaseUI.get_argument_names(RLHFArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_datasets', label=''):
            with gr.Blocks():
                DATA_SHOW.build_ui(DATA_SHOW)
                




