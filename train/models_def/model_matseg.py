import torch
import torch.nn as nn

from models_def import resnet_scene as resnet
import torch.nn.functional as F

def logit_embedding_to_instance(mat_notlight_mask_cpu, logits, embeddings, opt):
    h, w = logits.shape[2], logits.shape[3]
    instance_list = []
    num_mat_masks_list = []
    predict_segmentation_list = []
    

    for sample_idx, (logit_single, embedding_single) in enumerate(zip(logits, embeddings)):
        # prob_single = torch.sigmoid(logit_single)
        # prob_single = input_dict['mat_notlight_mask_cpu'][sample_idx].to(opt.device).float()
        prob_single = mat_notlight_mask_cpu[sample_idx].to(opt.device).float()

        # fast mean shift

        if opt.bin_mean_shift_device == 'cpu':
            prob_single, logit_single, embedding_single = prob_single.cpu(), logit_single.cpu(), embedding_single.cpu()
        segmentation, sampled_segmentation = opt.bin_mean_shift.test_forward(
            prob_single, embedding_single, mask_threshold=0.1)
        
        # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
        # we thus use avg_pool_2d to smooth the segmentation results
        b = segmentation.t().view(1, -1, h, w)
        pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
        b = pooling_b.view(-1, h*w).t()
        segmentation = b

        predict_segmentation = torch.argmax(segmentation, dim=1).view(h, w) # reshape to [h, w]: [0, 1, ..., len(matching)-1]; use this for mask pooling!!!!
        num_mat_masks = torch.max(predict_segmentation) + 1

        predict_segmentation[prob_single.squeeze() <= 0.1] = num_mat_masks # [h, w]: [0, 1, ..., len(matching)]; the last being the light mask

        predict_instance = torch.zeros((50, h, w), device=predict_segmentation.device)
        for i in range(num_mat_masks):
            seg = predict_segmentation == i
            predict_instance[i, :, :] = seg.reshape(h, w) # segmentation[0..num_mat_masks-1] for plane instances
        predict_instance = predict_instance.long()

        instance_list.append(predict_instance)
        num_mat_masks_list.append(num_mat_masks)
        predict_segmentation_list.append(predict_segmentation)

    return torch.stack(instance_list), torch.stack(num_mat_masks_list), torch.stack(predict_segmentation_list)


        # # greedy match of predict segmentation and ground truth segmentation using cross entropy
        # # to better visualization
        # gt_plane_num = input_dict['num_mat_masks_batch'][sample_idx]
        # matching = match_segmentation(segmentation, prob_single.view(-1, 1), input_dict['instance'][sample_idx], gt_plane_num)

        # return cluster results
        # predict_segmentation = segmentation.cpu().numpy().argmax(axis=1) # reshape to [h, w]: [0, 1, ..., len(matching)-1]; use this for mask pooling!!!!



class ResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2

        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5


class Baseline(nn.Module):
    def __init__(self, cfg, embed_dims=2, input_dim=3):
        super(Baseline, self).__init__()

        orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained, input_dim=input_dim)
        self.backbone = ResNet(orig_resnet)

        self.relu = nn.ReLU(inplace=True)

        channel = 64
        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # # plane or non-plane classifier
        self.pred_prob = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # embedding
        self.embedding_conv = nn.Conv2d(channel, embed_dims, (1, 1), padding=0)
        # # depth prediction
        # self.pred_depth = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # # surface normal prediction
        # self.pred_surface_normal = nn.Conv2d(channel, 3, (1, 1), padding=0)
        # # surface plane parameters
        # self.pred_param = nn.Conv2d(channel, 3, (1, 1), padding=0)

    def top_down_original(self, x):
        c1, c2, c3, c4, c5 = x
        # torch.Size([8, 128, 120, 160]) torch.Size([8, 256, 60, 80]) torch.Size([8, 512, 30, 40]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 2048, 8, 10])
        # print(c1.shape, c2.shape, c3.shape, c4.shape, c5.shape)

        p5 = self.relu(self.c5_conv(c5))

        # --- torch.Size([8, 64, 8, 10]) torch.Size([8, 64, 16, 20]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 64, 15, 20]) ------
        # print('---', p5.shape, self.upsample(p5).shape, c4.shape, self.c4_conv(c4).shape, '------')

        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4, p5

    def top_down(self, x):
        c1, c2, c3, c4, c5 = x
        # torch.Size([8, 128, 120, 160]) torch.Size([8, 256, 60, 80]) torch.Size([8, 512, 30, 40]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 2048, 8, 10])
        # print(c1.shape, c2.shape, c3.shape, c4.shape, c5.shape)

        p5 = self.relu(self.c5_conv(c5))

        # --- torch.Size([8, 64, 8, 10]) torch.Size([8, 64, 16, 20]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 64, 15, 20]) ------
        # print('---', p5.shape, self.upsample(p5).shape, c4.shape, self.c4_conv(c4).shape, '------')

        # p4 = self.up_conv5(self.upsample(p5))
        # p4 += F.interpolate(self.relu(self.c4_conv(c4)), (p4.shape[2], p4.shape[3]), mode='bilinear', align_corners=True)
        # p3 = self.up_conv4(self.upsample(p4))
        # p3 += F.interpolate(self.relu(self.c3_conv(c3)), (p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        # p2 = self.up_conv3(self.upsample(p3))
        # p2 += F.interpolate(self.relu(self.c2_conv(c2)), (p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)
        # p1 = self.up_conv2(self.upsample(p2))
        # p1 += F.interpolate(self.relu(self.c1_conv(c1)), (p1.shape[2], p1.shape[3]), mode='bilinear', align_corners=True)

        p4 = self.relu(self.c4_conv(c4))
        p4 = p4 + F.interpolate(self.up_conv5(self.upsample(p5)), (p4.shape[2], p4.shape[3]), mode='bilinear', align_corners=True)
        p3 = self.relu(self.c3_conv(c3))
        p3 = p3 + F.interpolate(self.up_conv4(self.upsample(p4)), (p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p2 = self.relu(self.c2_conv(c2))
        p2 = p2 + F.interpolate(self.up_conv3(self.upsample(p3)), (p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)
        p1 = self.relu(self.c1_conv(c1))
        p1 = p1 + F.interpolate(self.up_conv2(self.upsample(p2)), (p1.shape[2], p1.shape[3]), mode='bilinear', align_corners=True)


        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c1, c2, c3, c4, c5 = self.backbone(x)
        # print(x.shape, c1.shape, c2.shape, c3.shape, c4.shape, c5.shape) # torch.Size([8, 3, 240, 320]) torch.Size([8, 128, 120, 160]) torch.Size([8, 256, 60, 80]) torch.Size([8, 512, 30, 40]) torch.Size([8, 1024, 15, 20]) torch.Size([8, 2048, 8, 10])

        # top down
        p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5)) # [16, 3, 192, 256],  [16, 64, 96, 128],  [16, 128, 48, 64],  [16, 256, 24, 32],  [16, 256, 12, 16],  [16, 512, 6, 8]
        feats_matseg_dict = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
        # print(p0.shape, p1.shape, p2.shape, p3.shape, p4.shape, p5.shape)

        # output
        logit = self.pred_prob(p0)
        embedding = self.embedding_conv(p0)
        # depth = self.pred_depth(p0)
        # surface_normal = self.pred_surface_normal(p0)
        # param = self.pred_param(p0)

        return_dict = {'logit': logit, 'embedding': embedding, 'feats_matseg_dict': feats_matseg_dict}
        return return_dict
        # return prob, embedding
        # return prob, embedding, depth, surface_normal, param
