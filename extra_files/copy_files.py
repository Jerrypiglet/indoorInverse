from pathlib import Path
import shutil

data_root_slow = Path('/newfoundland2/ruizhu/DatasetNew_test')
data_root_fast = Path('/ruidata/openrooms_raw_BRDF')
modalities_list = [
    # ('im', '.rgbe'), 
    ('im_', '.hdr'), 
    ('imbaseColor_', '.png'), 
    ('imnormal_', '.png'), 
    ('imroughness_', '.png'), 
    ('immask_', '.png'), 
    ('imdepth_', '.dat'), 
    # ('immetallic_', '.png'), 
    # ('imenv_', '.hdr'), 
    # ('imcadmatobj_', '.dat'), 
    # ('imenvDirect_', '.hdr'), 
    # ('imDirect_', '.hdr'), 
]

for meta_split in ['mainDiffMat_xml', 'main_xml1', 'mainDiffMat_xml1', 'mainDiffLight_xml', 'mainDiffLight_xml1', 'main_xml']:
    for scene_name in [_.name for _ in (data_root_slow/meta_split).iterdir() if 'scene' in _.name]:
        ori_scene_path = data_root_slow / meta_split / scene_name
        dest_scene_path = data_root_fast / meta_split / scene_name
        for file in ori_scene_path.iterdir():
            for (name_start, name_end) in modalities_list:
                filename = file.name
                if filename.startswith(name_start) and filename.endswith(name_end):
                    file_ori = str(file)
                    file_dest = file_ori.replace(str(data_root_slow), str(data_root_fast))
                    if not dest_scene_path.exists():
                        dest_scene_path.mkdir(exist_ok=True, parents=True)
                    # if not Path(file_dest).exists():
                    shutil.copy2(file_ori, file_dest)
                    print(file_ori, file_dest)




