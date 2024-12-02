import os
from swift.ui.llm_datasets.show_data import base_path

OPENBAYES_DATASET_MAPPING = []

dirs = os.listdir(base_path)

for dir in dirs:
    dataset_path = os.path.join(base_path,dir)
    for names in os.listdir(dataset_path):
        if names == 'origin' or 'output' in names:
            sel_path = os.path.join(dataset_path,names)
            if os.path.isdir(sel_path):
                files = os.listdir(sel_path)
                for file in files:
                    if file.split('.')[-1] == 'json' or file.split('.')[-1] == 'jsonl' or file.split('.')[-1] == 'csv':
                        OPENBAYES_DATASET_MAPPING.append(os.path.join(sel_path,file))




