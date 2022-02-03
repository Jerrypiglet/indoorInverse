from pathlib import Path
from tqdm import tqdm
from icecream import ic
import shutil

folders = ['Checkpoint',]

ckpt_remove_list = []
ckpt_keep_list = []

for folder in folders:
    log_paths = Path(folder).iterdir()
    # print([x.name for x in log_paths])
    for log_path in log_paths:
        task_name = log_path.name
        latest_ckpt_path = log_path / 'last_checkpoint'
        if not latest_ckpt_path.exists():
            continue
        with open(latest_ckpt_path) as f:
            last_ckpt_path = f.read().splitlines()[0]
        last_ckpt_file_name = last_ckpt_path.split('/')[1]
        latest_ckpt_path = log_path / last_ckpt_file_name

        ckpts = [str(x) for x in Path(log_path).iterdir() if 'checkpointer_' in str(x)]
        assert str(latest_ckpt_path) in ckpts

        for ckpt in ckpts:
            if str(latest_ckpt_path) != ckpt:
                print('Delete ' + ckpt)
                ckpt_remove_list.append(Path(ckpt))
            else:
                print('Keep ' + ckpt)
                ckpt_keep_list.append(Path(ckpt))
        # assert latest_ckpt_path in ckpts
    
if_delete = input('%d ckpts to keep and %d to remove. Confirm? [y/n]'%(len(ckpt_keep_list), len(ckpt_remove_list)))

if if_delete == 'y':
    for ckpt in tqdm(ckpt_remove_list):
        ckpt.unlink()



        # if task_name.endswith('--tmp') or task_name.endswith('--tmp_new'):
        #     shutil.rmtree(log_path, ignore_errors=True)
        #     print('Removed '+str(log_path))
        #     continue

        # for task_datetime in mylist:
        #     if len(task_datetime.split('-'))==2:
        #         restore_task_datetime = task_datetime
        #     elif len(task_datetime.split('-'))==6:
        #         restore_task_datetime = '-'.join([task_datetime.split('-')[3].replace('gpu', ''), task_datetime.split('-')[4]])
        #     if task_name.startswith(restore_task_datetime) :
        #         # Path(log_path).unlink()
        #         shutil.rmtree(log_path, ignore_errors=True)
        #         print('Removed '+str(log_path))

