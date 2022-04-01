import os.path as osp
import cv2
import numpy as np
# import trimesh
from PIL import Image, ImageDraw, ImageFont
import os.path as osp

def loadImage(imName, isGama = False):
    imName = str(imName)
    if not(osp.isfile(imName ) ):
        assert False, imName

    im = Image.open(imName)
    # im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS )

    im = np.asarray(im, dtype=np.float32)
    if isGama:
        im = (im / 255.0) ** 2.2
        im = 2 * im - 1
    else:
        im = (im - 127.5) / 127.5
    if len(im.shape) == 2:
        im = im[:, np.newaxis]
    im = np.transpose(im, [2, 0, 1] )

    return im

def load_OR_hdr_uint8(frame_hdr_path_ori, seg=None, scale=None, extra_scale=1.):
    im_hdr = loadHdr(str(frame_hdr_path_ori))
    if seg is None and scale is None:
        seg_file = str(frame_hdr_path_ori).replace('im_', 'immask_').replace('hdr', 'png').replace('DiffMat', '')
        seg = 0.5 * (loadImage(seg_file) + 1)[0, :, :]
        seg = seg[:, :, np.newaxis]
    im, scale = scaleHdr(im_hdr, seg=seg, scale=scale)
    im = im * extra_scale
    im_not_hdr = np.clip(im**(1.0/2.2), 0., 1.)
    im_uint8 = (255. * im_not_hdr).astype(np.uint8)
    return im_uint8, (im_hdr, seg, scale)


def loadHdr(imName, if_resize=False, imWidth=None, imHeight=None, if_channel_first=False):
    imName = str(imName)
    if not(osp.isfile(imName ) ):
        print(imName )
        assert(False ), imName + ' NOT EXIST!'
    im = cv2.imread(imName, -1)
    # print(imName, im.shape, im.dtype)

    if im is None:
        print(imName )
        assert(False )

    if if_resize:
        im = cv2.resize(im, (imWidth, imHeight), interpolation = cv2.INTER_AREA )
    im = im[:, :, ::-1]
    if if_channel_first:
        im = np.transpose(im, [2, 0, 1])
    return im

def loadHdr_simple(imName):
    im_rec = cv2.imread(str(imName), -1 )
    print(imName, np.amax(im_rec))
    im_rec = np.ascontiguousarray(im_rec[:, :, ::-1] )
    return im_rec

def scaleHdr(hdr, seg, scale=None):
    if scale is None:
        imHeight, imWidth = hdr.shape[:2]
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * imWidth * imHeight * 3) ], 0.1, None)
    hdr = scale * hdr
    return np.clip(hdr, 0, 1), scale 

def to_nonhdr(im, scale=None, extra_scale=1.):
    if scale is None:
        seg = np.amin(im, 2)[:, :, np.newaxis] > 0.
        im, scale = scaleHdr(im, seg, scale=scale)
    else:
        seg = None
        im = im * scale
    im = im * extra_scale
    im_not_hdr = np.clip((im)**(1.0/2.2), 0., 1.)
    im_uint8 = (255. * im_not_hdr).astype(np.uint8)
    return im_uint8, scale

def in_frame(p, width, height):
    if p[0]>0 and p[0]<width and p[1]>0 and p[1]<height:
        return True
    else:
        return False

# from utils.utils_rui import clip
# def clip2rec(polygon, W, H, line_width=5):
#     # if not fix_polygon:
#     #     return polygon
#     if all_outside_rect(polygon, W, H):
#         return []
#     rectangle = [(-line_width, -line_width), (W+line_width, -line_width), (W+line_width, H+line_width), (-line_width, H+line_width)]
#     return clip(polygon, rectangle)

# def all_outside_rect(polygon, W, H):
#     if all([x[0] < 0 or x[0] >= W or x[1] < 0 or x[1] >= H for x in polygon]):
#         return True
#     else:
#         return False

def draw_lines_notalloutside_image(draw, v_list, idx_list, front_flags, color=(255, 255, 255), width=5):
    assert len(v_list) == len(front_flags)
    for i in range(len(idx_list)-1):
        if front_flags[idx_list[i]] and front_flags[idx_list[i+1]]:
            draw.line([v_list[idx_list[i]], v_list[idx_list[i+1]]], width=width, fill=color)

def draw_projected_bdb3d(draw, bdb2D_from_3D, front_flags=None, color=(255, 255, 255), width=5):
    bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]
    if front_flags is None:
        front_flags = [True] * len(bdb2D_from_3D)
    assert len(front_flags) == len(bdb2D_from_3D)

    for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        draw_lines_notalloutside_image(draw, bdb2D_from_3D, idx_list, front_flags, color=color, width=width)

    # W, H = img_map.size

    # print(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]], W=W, H=H, line_width=width))

    # draw.line(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[1], bdb2D_from_3D[2], bdb2D_from_3D[3], bdb2D_from_3D[0]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[4], bdb2D_from_3D[5], bdb2D_from_3D[6], bdb2D_from_3D[7], bdb2D_from_3D[4]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[0], bdb2D_from_3D[4]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[1], bdb2D_from_3D[5]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[2], bdb2D_from_3D[6]], W=W, H=H, line_width=width),
    #     fill=color, width=width)
    # draw.line(clip2rec([bdb2D_from_3D[3], bdb2D_from_3D[7]], W=W, H=H, line_width=width),
    #     fill=color, width=width)