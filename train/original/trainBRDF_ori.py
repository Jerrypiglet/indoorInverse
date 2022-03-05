import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils_ori as utils
import dataLoader_ori as dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from tqdm import tqdm
import time
from pathlib import Path

from train_funcs_brdf_ori_adapted_ import train_step, val_epoch, get_input_batch

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='/ruidata/openrooms_raw_BRDF', help='path to input images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--nepoch1', type=int, default=10, help='the number of epochs for training')

parser.add_argument('--batchSize0', type=int, default=16, help='input batch size')
parser.add_argument('--batchSize1', type=int, default=16, help='input batch size')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training network')
# Fine tune the network
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the network')
parser.add_argument('--epochIdFineTune', type=int, default = 0, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')   

# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# Rui
parser.add_argument('--ifMatMapInput', action='store_true', help='using mask as additional input')
parser.add_argument('--ifDataloaderOnly', action='store_true', help='benchmark dataloading overhead')
parser.add_argument('--ifCluster', action='store_true', help='if using cluster')
parser.add_argument('--eval_every_iter', type=int, default=2000, help='the casacade level')


# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

opt.albeW, opt.normW = opt.albedoWeight, opt.normalWeight
opt.rougW = opt.roughWeight
opt.deptW = opt.depthWeight

if opt.cascadeLevel == 0:
    opt.nepoch = opt.nepoch0
    opt.batchSize = opt.batchSize0
    opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0
elif opt.cascadeLevel == 1:
    opt.nepoch = opt.nepoch1
    opt.batchSize = opt.batchSize1
    opt.imHeight, opt.imWidth = opt.imHeight1, opt.imWidth1

if opt.experiment is None:
    opt.experiment = 'check_cascade%d_w%d_h%d' % (opt.cascadeLevel,
            opt.imWidth, opt.imHeight )
opt.experiment = 'logs/' + opt.experiment
if opt.ifCluster:
    opt.experiment = '/viscompfs/users/ruizhu/' + opt.experiment
os.system('rm -rf {0}'.format(opt.experiment) )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp -r train %s' % opt.experiment )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initial Network
# encoder = models.encoder0(cascadeLevel = opt.cascadeLevel, in_channels = 3 if not opt.ifMatMapInput else 4)
encoder = models.encoder0(cascadeLevel = opt.cascadeLevel)
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )
####################################################################


#########################################
lr_scale = 1
if opt.isFineTune:
    print('--- isFineTune=True')
    encoder.load_state_dict(
            torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
    albedoDecoder.load_state_dict(
            torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
    normalDecoder.load_state_dict(
            torch.load('{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
    roughDecoder.load_state_dict(
            torch.load('{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
    depthDecoder.load_state_dict(
            torch.load('{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.epochIdFineTune) ).module.state_dict() )
    lr_scale = 1.0 / (2.0 ** (np.floor( ( (opt.epochIdFineTune+1) / 10)  ) ) )
else:
    opt.epochIdFineTune = -1
#########################################
model = {}
model['encoder'] = nn.DataParallel(encoder, device_ids = opt.deviceIds )
model['albedoDecoder'] = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
model['normalDecoder'] = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
model['roughDecoder'] = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
model['depthDecoder'] = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

##############  ######################
# Send things into GPU
if opt.cuda:
    model['encoder'] = model['encoder'].cuda(opt.gpuId )
    model['albedoDecoder'] = model['albedoDecoder'].cuda()
    model['normalDecoder'] = model['normalDecoder'].cuda()
    model['roughDecoder'] = model['roughDecoder'].cuda()
    model['depthDecoder'] = model['depthDecoder'].cuda()
####################################


####################################
# Optimizer
optimizer = {}
optimizer['opEncoder'] = optim.Adam(model['encoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
optimizer['opAlbedo'] = optim.Adam(model['albedoDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
optimizer['opNormal'] = optim.Adam(model['normalDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
optimizer['opRough'] = optim.Adam(model['roughDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
optimizer['opDepth'] = optim.Adam(model['depthDecoder'].parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
#####################################


####################################
brdfDatasetTrain = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight,
        cascadeLevel = opt.cascadeLevel, split = 'train')
brdfLoaderTrain = DataLoader(brdfDatasetTrain, batch_size = opt.batchSize,
        num_workers = 16, shuffle = True, pin_memory=True)

brdfDatasetVal = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight,
        cascadeLevel = opt.cascadeLevel, split = 'val')
brdfLoaderVal = DataLoader(brdfDatasetVal, batch_size = opt.batchSize,
        num_workers = 8, shuffle = False, pin_memory=True)

# writer = SummaryWriter(log_dir=str(Path('log') / opt.experiment), flush_secs=10) # relative path
writer = SummaryWriter(log_dir=opt.experiment, flush_secs=10) # relative path


j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

ts_iter_end_start_list = []
ts_iter_start_end_list = []

for epoch in list(range(opt.epochIdFineTune+1, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    ts_epoch_start = time.time()
    # ts = ts_epoch_start
    # ts_iter_start = ts
    ts_iter_end = ts_epoch_start

    for i, dataBatch in tqdm(enumerate(brdfLoaderTrain)):
        if j % opt.eval_every_iter == 0:
            val_epoch(brdfLoaderVal, model, optimizer, writer, opt, j)
            ts_iter_end = time.time()

        j += 1
        ts_iter_start = time.time()
        if j > 5:
            ts_iter_end_start_list.append(ts_iter_start - ts_iter_end)

        if opt.ifDataloaderOnly:
            continue

        
        input_batch, inputDict, preBatchDict = get_input_batch(dataBatch, opt)

        errors = train_step(input_batch, inputDict, preBatchDict, optimizer, model, opt)

        # Output training error
        utils.writeErrToScreen('albedo', errors['albedoErrs'], epoch, j )
        utils.writeErrToScreen('normal', errors['normalErrs'], epoch, j )
        utils.writeErrToScreen('rough', errors['roughErrs'], epoch, j )
        utils.writeErrToScreen('depth', errors['depthErrs'], epoch, j )
        
        writer.add_scalar('loss_train/loss_albedo', errors['albedoErrs'][0].item(), j)
        writer.add_scalar('loss_train/loss_normal', errors['normalErrs'][0].item(), j)
        writer.add_scalar('loss_train/loss_rough', errors['roughErrs'][0].item(), j)
        writer.add_scalar('loss_train/loss_depth', errors['depthErrs'][0].item(), j)

        utils.writeErrToFile('albedo', errors['albedoErrs'], trainingLog, epoch, j )
        utils.writeErrToFile('normal', errors['normalErrs'], trainingLog, epoch, j )
        utils.writeErrToFile('rough', errors['roughErrs'], trainingLog, epoch, j )
        utils.writeErrToFile('depth', errors['depthErrs'], trainingLog, epoch, j )

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(errors['albedoErrs'])], axis=0)
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(errors['normalErrs'])], axis=0)
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(errors['roughErrs'])], axis=0)
        depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(errors['depthErrs'])], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)


        if j == 1 or j% 2000 == 0:
            # Save the ground truth and the input
            vutils.save_image(( (inputDict['albedoBatch'] ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoGt.png'.format(opt.experiment, j) )
            vutils.save_image( (0.5*(inputDict['normalBatch'] + 1) ).data,
                    '{0}/{1}_normalGt.png'.format(opt.experiment, j) )
            vutils.save_image( (0.5*(inputDict['roughBatch'] + 1) ).data,
                    '{0}/{1}_roughGt.png'.format(opt.experiment, j) )
            vutils.save_image( ( (inputDict['imBatch'])**(1.0/2.2) ).data,
                    '{0}/{1}_im.png'.format(opt.experiment, j) )
            depthOut = 1 / torch.clamp(inputDict['depthBatch'] + 1, 1e-6, 10) * inputDict['segAllBatch'].expand_as(inputDict['depthBatch'])
            vutils.save_image( ( depthOut*inputDict['segAllBatch'].expand_as(inputDict['depthBatch']) ).data,
                    '{0}/{1}_depthGt.png'.format(opt.experiment, j) )

            if opt.cascadeLevel > 0:
                vutils.save_image( ( (preBatchDict['diffusePreBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_diffusePre.png'.format(opt.experiment, j) )
                vutils.save_image( ( (preBatchDict['specularPreBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_specularPre.png'.format(opt.experiment, j) )
                vutils.save_image( ( (preBatchDict['renderedImBatch'])**(1.0/2.2) ).data,
                        '{0}/{1}_renderedImage.png'.format(opt.experiment, j) )

            # Save the predicted results
            for n in range(0, len(preBatchDict['albedoPreds']) ):
                vutils.save_image( ( (preBatchDict['albedoPreds'][n] ) ** (1.0/2.2) ).data,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
            for n in range(0, len(preBatchDict['normalPreds']) ):
                vutils.save_image( ( 0.5*(preBatchDict['normalPreds'][n] + 1) ).data,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
            for n in range(0, len(preBatchDict['roughPreds']) ):
                vutils.save_image( ( 0.5*(preBatchDict['roughPreds'][n] + 1) ).data,
                        '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
            for n in range(0, len(preBatchDict['depthPreds']) ):
                depthOut = 1 / torch.clamp(preBatchDict['depthPreds'][n] + 1, 1e-6, 10) * inputDict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n])
                vutils.save_image( ( depthOut * inputDict['segAllBatch'].expand_as(preBatchDict['depthPreds'][n]) ).data,
                        '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

        ts_iter_end = time.time()
        if j > 5:
            ts_iter_start_end_list.append(ts_iter_end - ts_iter_start)
            # if j % 10 == 0:
            print('Rolling end-to-start %.2f, Rolling start-to-end %.2f'%(sum(ts_iter_end_start_list)/len(ts_iter_end_start_list), sum(ts_iter_start_end_list)/len(ts_iter_start_end_list)))
            # print(ts_iter_end_start_list, ts_iter_start_end_list)

        writer.add_scalar('training/epoch', epoch, j)



    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 10 == 0:
        for param_group in optimizer['opEncoder'].param_groups:
            param_group['lr'] /= 2
        for param_group in optimizer['opAlbedo'].param_groups:
            param_group['lr'] /= 2
        for param_group in optimizer['opNormal'].param_groups:
            param_group['lr'] /= 2
        for param_group in optimizer['opRough'].param_groups:
            param_group['lr'] /= 2
        for param_group in optimizer['opDepth'].param_groups:
            param_group['lr'] /= 2
    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )

    # save the models
    torch.save(model['encoder'].module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(model['albedoDecoder'].module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(model['normalDecoder'].module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(model['roughDecoder'].module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(model['depthDecoder'].module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
