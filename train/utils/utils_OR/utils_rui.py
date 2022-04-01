from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

def clip(subjectPolygon, clipPolygon):
   # https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0]
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return ((n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3)

   outputList = subjectPolygon
   cp1 = clipPolygon[-1]

   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]

      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

def vis_cube_plt(Xs, ax, color=None, linestyle='-', label=None, if_face_idx_text=False, if_vertex_idx_text=False, text_shift=[0., 0., 0.], fontsize_scale=1., linewidth=1.):
   # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
   index1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
   index2 = [[1, 5], [2, 6], [3, 7]]
   index3 = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]] # faces
   if color is None:
      color = list(np.random.choice(range(256), size=3) / 255.)
   #   print(color)
   ax.plot3D(Xs[index1, 0], Xs[index1, 1], Xs[index1, 2], color=color, linestyle=linestyle, linewidth=linewidth)
   for index in index2:
      ax.plot3D(Xs[index, 0], Xs[index, 1], Xs[index, 2], color=color, linestyle=linestyle, linewidth=linewidth)
   if label is not None:
      # ax.text3D(Xs[0, 0]+text_shift[0], Xs[0, 1]+text_shift[1], Xs[0, 2]+text_shift[2], label, color=color, fontsize=10*fontsize_scale)
      ax.text3D(Xs.mean(axis=0)[0], Xs.mean(axis=0)[1], Xs.mean(axis=0)[2], label, color=color, fontsize=10*fontsize_scale)
   if if_vertex_idx_text:
      for vertex_idx, V in enumerate(Xs):
         ax.text3D(V[0]+text_shift[0], V[1]+text_shift[1], V[2]+text_shift[2], str(vertex_idx), color=color, fontsize=10*fontsize_scale)
   if if_face_idx_text:
      for face_idx, index in enumerate(index3):
         X_center = Xs[index, :].mean(0)
         # print(X_center.shape)
         ax.text3D(X_center[0]+text_shift[0], X_center[1]+text_shift[1], X_center[2]+text_shift[2], str(face_idx), color='grey', fontsize=30*fontsize_scale)
         ax.scatter3D(X_center[0], X_center[1], X_center[2], color=[0.8, 0.8, 0.8], s=30)
         for cross_index in [[index[0], index[2]], [index[1], index[3]]]:
            ax.plot3D(Xs[cross_index, 0], Xs[cross_index, 1], Xs[cross_index, 2], color=[0.8, 0.8, 0.8], linestyle='--', linewidth=1)


   

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def vis_axis(ax):
    for vec, tag, tag_loc in zip([([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]), ([0, 0], [0, 0], [0, 1])], [r'$X_w$', r'$Y_w$', r'$Z_w$'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color="k")
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag)
        ax.add_artist(a)

def vis_axis_xyz(ax, x, y, z, origin=[0., 0., 0.], suffix='_w', color='k'):
    for vec, tag, tag_loc in zip([([origin[0], (origin+x)[0]], [origin[1], (origin+x)[1]], [origin[2], (origin+x)[2]]), \
       ([origin[0], (origin+y)[0]], [origin[1], (origin+y)[1]], [origin[2], (origin+y)[2]]), \
          ([origin[0], (origin+z)[0]], [origin[1], (origin+z)[1]], [origin[2], (origin+z)[2]])], [r'$X%s$'%suffix, r'$Y%s$'%suffix, r'$Z%s$'%suffix], [origin+x, origin+y, origin+z]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                lw=1, arrowstyle="->", color=color)
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag, color=color)
        ax.add_artist(a)

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

   
import numpy as np
from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    H = np.linalg.norm(rval[0]-rval[1])
    W = np.linalg.norm(rval[1]-rval[2])
    return rval, (H, W), np.min(areas)