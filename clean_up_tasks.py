from pathlib import Path
from tqdm import tqdm
from icecream import ic
import shutil

list_path = 'clean_up_tasks.txt'
black_list_path = 'clean_up_tasks_black.txt'

folders = ['Checkpoint', 'logs', 'Summary_vis']

with open(list_path) as f:
    mylist = f.read().splitlines() 
mylist = [x.strip() for x in mylist]

with open(black_list_path) as f:
    black_mylist = f.read().splitlines() 
black_mylist = [x.strip().replace('zz-torch-job-gpu', '').split('--')[0] for x in black_mylist]

for folder in folders:
    log_paths = Path(folder).iterdir()
    # print([x.name for x in log_paths])
    for log_path in log_paths:
        task_name = log_path.name
        if task_name.startswith('tmp') or task_name.endswith('--tmp') or task_name.endswith('--tmp_new') or task_name in black_mylist or task_name.split('--')[0] in black_mylist:
            shutil.rmtree(log_path, ignore_errors=True)
            print('Removed '+str(log_path))
            continue
        
        for task_datetime in mylist:
            if len(task_datetime.split('-'))>=2 and len(task_datetime.split('-'))<6:
                restore_task_datetime = task_datetime
            elif len(task_datetime.split('-'))==6:
                restore_task_datetime = '-'.join([task_datetime.split('-')[3].replace('gpu', ''), task_datetime.split('-')[4]])
            if task_name.startswith(restore_task_datetime) :
                # Path(log_path).unlink()
                shutil.rmtree(log_path, ignore_errors=True)
                print('Removed '+str(log_path))

