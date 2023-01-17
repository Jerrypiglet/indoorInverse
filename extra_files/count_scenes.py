# list_path = 'train/data/openrooms/list_OR_V4full_CVPR20/list/train.txt' # 1175 scenes, for training in Li'20
# list_path = 'train/data/openrooms/list_OR_V4full_CVPR20/list/val.txt' # 109 scenes, for testing in Li'20

# list_path = 'train/data/openrooms/list_OR_V4full/list/train.txt' # 6684 scenes; for training in Zhu'22
# list_path = 'train/data/openrooms/list_OR_V4full/list/val.txt' # 354 scenes
# list_path = 'train/data/openrooms/list_OR_V4full/list/test.txt' # 654 scenes
# list_path = 'train/data/openrooms/list_OR_V4full/list/valtest.txt' # 1008 scenes; for testing in Zhu'22

splits = ['train', 'valtest', 'val', 'test']; list_path = 'train/data/openrooms/list_OR_V4full/list/%s.txt'
if_dump_scene_list = False

scene_list_dict = {}
scene_name_list_dict = {_: [] for _ in splits}
for split in splits:
    list_read_scenes = [_.strip() for _ in open(list_path%split).readlines()]
    list_read_scenes = ['/'.join(_.split(' ')[2].split('/')[:2]) for _ in list_read_scenes] # ONLY for IRISFormer splits; comment this line if count list_OR_V4full_CVPR20 scenes
    list_read_scenes = list(set(list_read_scenes))

    scene_list_dict[split] = list_read_scenes
    scene_name_list_dict[split] = [_.split('/')[1] for _ in list_read_scenes]

    print('====', split, len(scene_list_dict[split]))
    print(scene_list_dict[split][:2])
    print(scene_name_list_dict[split][:2])

    if if_dump_scene_list:
        dump_path1 = (list_path%split).replace('.txt', '_scenes.txt')
        dump_path2 = (list_path%split).replace('.txt', '_scene_names.txt')
        with open(dump_path1, 'w') as f:
            for line in scene_list_dict[split]:
                f.write(f"{line}\n")
        with open(dump_path2, 'w') as f:
            for line in scene_name_list_dict[split]:
                f.write(f"{line}\n")

    # from pathlib import Path
    # data_root_slow = Path('/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/DatasetCreation')
    # data_root_slow = Path('/newfoundland2/ruizhu/DatasetNew_test')
    
    # for scene in list_read_scenes:
    #     meta_split, scene_name = scene.split('/')
    #     scene_path = data_root_slow / meta_split / scene_name
    #     if not (scene_path / ('im_1.hdr')).exists():
    #         print(meta_split, scene_name)

print('==== overlap')
print([_ for _ in scene_list_dict['valtest'] if _ in scene_list_dict['train']])
print([_ for _ in scene_name_list_dict['valtest'] if _ in scene_name_list_dict['train']])

print([_ for _ in scene_list_dict['val'] if _ in scene_list_dict['train']])
print([_ for _ in scene_name_list_dict['val'] if _ in scene_name_list_dict['train']])

print([_ for _ in scene_list_dict['test'] if _ in scene_list_dict['train']])
print([_ for _ in scene_name_list_dict['test'] if _ in scene_name_list_dict['train']])
