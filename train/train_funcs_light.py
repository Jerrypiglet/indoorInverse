import torch
from torch.autograd import Variable
# import models
import torch.nn.functional as F
from tqdm import tqdm
import statistics
import torchvision.utils as vutils

from train_funcs_brdf import get_labels_dict_brdf

def get_labels_dict_light(data_batch, opt, list_from_brdf=None, return_input_batch_as_list=True):

    if list_from_brdf is None:
        input_batch, input_dict, preBatchDict = get_labels_dict_brdf(data_batch, opt, return_input_batch_as_list=True)
    else:
        input_batch, input_dict, preBatchDict = list_from_brdf
        
    extra_dict = {}

    if opt.cfg.DATA.load_light_gt:
        envmaps_cpu = data_batch['envmaps']
        envmapsBatch = Variable(envmaps_cpu ).cuda(non_blocking=True)

        hdr_scale_cpu = data_batch['hdr_scale']
        hdr_scaleBatch = Variable(hdr_scale_cpu ).cuda(non_blocking=True)

        envmapsInd_cpu = data_batch['envmapsInd']
        envmapsIndBatch = Variable(envmapsInd_cpu ).cuda(non_blocking=True)

        extra_dict.update({'envmapsBatch': envmapsBatch, 'envmapsIndBatch': envmapsIndBatch, 'hdr_scaleBatch': hdr_scaleBatch})

        if opt.cfg.MODEL_LIGHT.load_GT_light_sg:
            extra_dict.update({'sg_theta_Batch': data_batch['sg_theta'].cuda(non_blocking=True), 'sg_phi_Batch': data_batch['sg_phi'].cuda(non_blocking=True)})
            extra_dict.update({'sg_axis_Batch': data_batch['sg_axis'].cuda(non_blocking=True)})
            extra_dict.update({'sg_lamb_Batch': data_batch['sg_lamb'].cuda(non_blocking=True)})
            extra_dict.update({'sg_weight_Batch': data_batch['sg_weight'].cuda(non_blocking=True)})

        if opt.cascadeLevel > 0:

            diffusePre_cpu = data_batch['diffusePre']
            diffusePreBatch = Variable(diffusePre_cpu ).cuda(non_blocking=True)

            specularPre_cpu = data_batch['specularPre']
            specularPreBatch = Variable(specularPre_cpu ).cuda(non_blocking=True)

            # Regress the diffusePred and specular Pred
            envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
            imBatchSmall = F.adaptive_avg_pool2d(input_dict['imBatch'], (envRow, envCol) )
            diffusePreBatch, specularPreBatch = models.LSregressDiffSpec(
                    diffusePreBatch, specularPreBatch, imBatchSmall,
                    diffusePreBatch, specularPreBatch )

            if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
                diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
            if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
                specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

            renderedImBatch = diffusePreBatch + specularPreBatch

            input_batch += [diffusePreBatch, specularPreBatch]

            preBatchDict.update({'diffusePreBatch': diffusePreBatch, 'specularPreBatch': specularPreBatch})
            preBatchDict['renderedImBatch'] = renderedImBatch

            envmapsPre_cpu = data_batch['envmapsPre']
            envmapsPreBatch = Variable(envmapsPre_cpu ).cuda(non_blocking=True)
            input_dict['envmapsPreBatch'] = envmapsPreBatch


    if not return_input_batch_as_list:
        input_batch = torch.cat(input_batch, dim=1)

    return input_batch, input_dict, preBatchDict, extra_dict

def postprocess_light(input_dict, output_dict, loss_dict, opt, time_meters):
    # Compute the recontructed error
    if not opt.cfg.DATASET.if_no_gt_light:
        if opt.cfg.MODEL_LIGHT.use_scale_aware_loss:
            reconstErr_loss_map = ( torch.log(output_dict['envmapsPredImage'] + opt.cfg.MODEL_LIGHT.offset) - torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) \
                * \
                    ( torch.log(output_dict['envmapsPredImage'] + opt.cfg.MODEL_LIGHT.offset ) - torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) \
                * \
                    output_dict['segEnvBatch'].expand_as(output_dict['envmapsPredImage'] )
        else:
            # a = torch.log(output_dict['envmapsPredScaledImage'] + opt.cfg.MODEL_LIGHT.offset)
            # b = torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset )
            # if opt.is_master:
            #     print('>>>>', torch.max(a), torch.min(a), torch.median(a))
            #     print('>---', torch.max(b), torch.min(b), torch.median(b))
            # reconstErr_loss_map = ( torch.log(output_dict['envmapsPredScaledImage'] + opt.cfg.MODEL_LIGHT.offset) - torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) \
            #     * \
            #         ( torch.log(output_dict['envmapsPredScaledImage'] + opt.cfg.MODEL_LIGHT.offset ) - torch.log(input_dict['envmapsBatch'] + opt.cfg.MODEL_LIGHT.offset ) ) \
            #     * \
            #         output_dict['segEnvBatch'].expand_as(output_dict['envmapsPredImage'] ) 

            reconstErr_loss_map = (output_dict['envmapsPredScaledImage_offset_log_'] - torch.log(input_dict['envmapsBatch']+opt.cfg.MODEL_LIGHT.offset) ) \
                * \
                    ( output_dict['envmapsPredScaledImage_offset_log_'] - torch.log(input_dict['envmapsBatch']+opt.cfg.MODEL_LIGHT.offset) ) \
                * \
                    output_dict['segEnvBatch'].expand_as(output_dict['envmapsPredImage'] ) 

        reconstErr = torch.sum(reconstErr_loss_map) \
            / output_dict['pixelNum_recon'] / 3.0 / opt.cfg.MODEL_LIGHT.envWidth / opt.cfg.MODEL_LIGHT.envHeight

        output_dict['reconstErr_loss_map'] = reconstErr_loss_map

        loss_dict['loss_light-reconstErr'] = reconstErr

        # Compute the rendered error
        renderErr = torch.sum( (output_dict['renderedImPred'] - output_dict['imBatchSmall'])
            * (output_dict['renderedImPred'] - output_dict['imBatchSmall']) * output_dict['segBRDFBatchSmall'].expand_as(output_dict['imBatchSmall'] )  ) \
            / output_dict['pixelNum_render'] / 3.0
        loss_dict['loss_light-renderErr'] = renderErr

        # print(opt.renderWeight, opt.reconstWeight)
        loss_dict['loss_light-ALL'] = opt.renderWeight * renderErr + opt.reconstWeight * reconstErr
    # loss_dict['loss_light-ALL'] = opt.renderWeight * renderErr

    # torch.Size([4, 3, 120, 160, 8, 16]) torch.Size([4, 3, 120, 160, 8, 16]) torch.Size([4, 3, 120, 160]) torch.Size([4, 3, 120, 160])
    # print(output_dict['envmapsPredScaledImage'].shape, input_dict['envmapsBatch'].shape, output_dict['renderedImPred'].shape, output_dict['imBatchSmall'].shape)
    # import pickle
    # reindexed_pickle_path = 'a.pkl'
    # sequence = {'envmapsPredScaledImage': output_dict['envmapsPredScaledImage'].detach().cpu().numpy(), \
    #     'envmapsBatch': input_dict['envmapsBatch'].detach().cpu().numpy(), \
    #     'renderedImPred': output_dict['renderedImPred'].detach().cpu().numpy(), 
    #     'imBatchSmall': output_dict['imBatchSmall'].detach().cpu().numpy()}
    # with open(reindexed_pickle_path, 'wb') as f:
    #     pickle.dump(sequence, f, protocol=pickle.HIGHEST_PROTOCOL)



    return output_dict, loss_dict