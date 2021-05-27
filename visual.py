import numpy as np
import cv2 as cv
import numpy as np
import math
from config import config
from post_proc import get_keypoints
import matplotlib.pyplot as plt 
import matplotlib

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.:
        out = out / 255.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out

def visualize_short_offsets(offsets, keypoint_id, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None, every=1,save_path='./'):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            keypoint_id = config.KEYPOINTS.index(keypoint_id)
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==keypoint_id]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*keypoint_id:2*keypoint_id+2]
    masks = np.zeros(offsets.shape[:2]+(len(centers),), dtype='bool')
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists = np.sqrt(np.square(idx-c).sum(axis=-1))
        dists_x = np.abs(idx[:,:,0] - c[0])
        dists_y = np.abs(idx[:,:,1] - c[1])
        masks[:,:,j] = (dists<=radius)
        if every > 1:
            d_mask = np.logical_and(np.mod(dists_x.astype('int32'), every)==0, np.mod(dists_y.astype('int32'), every)==0)
            masks[:,:,j] = np.logical_and(masks[:,:,j], d_mask)
    mask = masks.sum(axis=-1) > 0
    
#     for j, c in enumerate(centers):
#         dists[:,:,j] = np.sqrt(np.square(idx-c).sum(axis=-1))
#     dists = dists.min(axis=-1)
#     mask = dists <= radius
    I, J = np.nonzero(mask)

    fig = plt.figure()
    if img is not None:
        plt.imshow(img)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.close()

    return figure_to_array(fig)
