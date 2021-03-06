import numpy as np
import os.path as osp
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
# print(sys.path)
from utils_OR.utils_OR_geo import isect_line_plane_v3

def read_cam_params(camFile):
    assert osp.isfile(str(camFile))
    with open(str(camFile), 'r') as camIn:
    #     camNum = int(camIn.readline().strip() )
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def normalize(x):
    return x / np.linalg.norm(x)

def project_v(v, cam_R, cam_t, cam_K, if_only_proj_front_v=False, if_return_front_flags=False, if_v_already_transformed=False, extra_transform_matrix=np.eye(3)):
    if if_v_already_transformed:
        v_transformed = v.T
    else:
        v_transformed = cam_R @ v.T + cam_t
    
    v_transformed = (v_transformed.T @ extra_transform_matrix).T
#     print(v_transformed[2:3, :])
    if if_only_proj_front_v:
        v_transformed = v_transformed * (v_transformed[2:3, :] > 0.)
    p = cam_K @ v_transformed
    if not if_return_front_flags:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, (v_transformed[2:3, :] > 0.).flatten().tolist()

def project_3d_line(x1x2, cam_R, cam_t, cam_K, cam_center, cam_zaxis, if_debug=False, extra_transform_matrix=np.eye(3)):
    assert len(x1x2.shape)==2 and x1x2.shape[1]==3
    # print(cam_R.shape, x1x2.T.shape, cam_t.shape)
    x1x2_transformed = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
    # print(x1x2_transformed)
    if if_debug:
        print('x1x2_transformed', x1x2_transformed)
    front_flags = list(x1x2_transformed[:, -1] > 0.)
    if if_debug:
        print('front_flags', front_flags)
    if not all(front_flags):
        if not front_flags[0] and not front_flags[1]:
            return None
        x_isect = isect_line_plane_v3(x1x2[0], x1x2[1], cam_center, cam_zaxis, epsilon=1e-6)
#             print(x1x2[front_flags.index(True)], x_isect)
        x1x2 = np.vstack((x1x2[front_flags.index(True)].reshape((1, 3)), x_isect.reshape((1, 3))))
        x1x2_transformed = (cam_R @ x1x2.T + cam_t).T @ extra_transform_matrix
        # print('-->', x1x2_transformed)
    if if_debug:
        print('x1x2_transformed after', x1x2_transformed)

    # x1x2_transformed = x1x2_transformed @ extra_transform_matrix
    # print(x1x2_transformed)
    p = cam_K @ x1x2_transformed.T
    if not if_debug:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T
    else:
        return np.vstack([p[0, :]/(p[2, :]+1e-8), p[1, :]/(p[2, :]+1e-8)]).T, x1x2

# def project_v_homo(v, cam_transformation4x4, cam_K):
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/img30.gif
#     # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
#     v_homo = np.hstack([v, np.ones((v.shape[0], 1))])
#     cam_K_homo = np.hstack([cam_K, np.zeros((3, 1))])
# #     v_transformed = cam_R @ v.T + cam_t

#     v_transformed = cam_transformation4x4 @ v_homo.T
#     v_transformed_nonhomo = np.vstack([v_transformed[0, :]/v_transformed[3, :], v_transformed[1, :]/v_transformed[3, :], v_transformed[2, :]/v_transformed[3, :]])
# #     print(v_transformed.shape, v_transformed_nonhomo.shape)
#     v_transformed = v_transformed * (v_transformed_nonhomo[2:3, :] > 0.)
#     p = cam_K_homo @ v_transformed
#     return np.vstack([p[0, :]/p[2, :], p[1, :]/p[2, :]]).T
