from utils.utils_misc import red
from train_funcs_joint_all import get_labels_dict_joint, forward_joint
from utils.comm import synchronize
from tqdm import tqdm

def dump_joint(brdf_loader_val, model, params_mis):
    pass
    writer, logger, opt, tid, batch_size = params_mis['writer'], params_mis['logger'], params_mis['opt'], params_mis['tid'], params_mis['batch_size_val_vis']
    logger.info(red('=== [vis_val_epoch_joint] Visualizing for %d batches on rank %d'%(len(brdf_loader_val), opt.rank)))

    model.eval()

    sample_list = []

    with torch.no_grad():
        for batch_id, data_batch in tqdm(enumerate(brdf_loader_val)):
            if batch_size*batch_id >= opt.cfg.TEST.vis_max_samples:
                break

            input_dict = get_labels_dict_joint(data_batch, opt)

            # ======= Forward
            output_dict, loss_dict = forward_joint(False, input_dict, model, opt, time_meters, if_vis=True)

            synchronize()

            for sample_idx_batch, (im_single, im_path) in enumerate(data_batch['image_path']):
                sample_idx = sample_idx_batch+batch_size*batch_id
                sample_dict = {}
                print('Dumping sample %d...'%sample_idx, batch_id, sample_idx_batch)
                if sample_idx >= opt.cfg.TEST.vis_max_samples:
                    break
                    
                im_single = data_batch['im_SDR_RGB'][sample_idx_batch]
                im_single = im_single.numpy().squeeze()


                sample_list.append(sample_dict)




