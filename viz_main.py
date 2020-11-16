import pathlib
import cv2
import os
import numpy as np
import mayavi.mlab as mlab
from PIL import Image
from objects import *
from viz_util import *
from draw_util import *

#idxlist = [159,182,212,235,248,1230,1585,2818,2858,2925,3283,3659,3804,3898,5381]
#idxlist = [2925,3283,3659,3804,3898,5381]
#idxlist = [143,2506,6531]
#idxlist = [143,159, 2506, 2818,3283,3804,5381,6156,6300,6531,6612,6752,7091]
#idxlist = [143,159, 2506,6531,6752,7091]
#idxlist =[2858]
#idxlist = [4064,4065,4068,4072,4074,4077,4079,4081,4082,4083,4085,4087,4089]
#idxlist = [2858,4064]
idxlist = [1,9,17,25,26,27,31,33,37,38,41,78,138,146,178,183,196,243,274,344]
rgb_dir = pathlib.Path('rgb')
label_dir = pathlib.Path('gt')
calib_dir = pathlib.Path('calib')
velo_dir = pathlib.Path('velo')
res_dir = pathlib.Path('results')
rgb_out = pathlib.Path('box2d')

def read_data(idx, read_gt=True):
    rgb = cv2.imread(rgb_dir.name +'/' + idx + '.png')
    if read_gt:
        objs = read_label(label_dir  / ( idx + '.txt'))
    else:
        objs = []
    velo = np.fromfile(velo_dir.name + '/' + idx + '.bin', dtype=np.float32).reshape(-1,4)
    res = read_label(res_dir  / ( idx + '.txt'))
    cal = Calibration(calib_dir  / ( idx + '.txt'))
    return idx, rgb, velo, objs, res, cal

def dataReader(read_data=read_data, read_gt=True):
    results = ['%06d.txt'%x for x in idxlist]
    results.sort()
    for r in results:
        yield read_data(r[:-4], read_gt)

if __name__=='__main__ draw lidar pc':
    l = np.fromfile('007480.bin', dtype=np.float32).reshape(-1,4)
    fig = mlab.figure(size=(1200, 800), bgcolor=(0.9, 0.9, 0.85))
    fig = draw_lidar_pc(l, fig=fig)
    mlab.show()

if __name__=='__main__':# draw boxes in 3d space':
    reader = dataReader(read_gt=False)
    draw_lidar_3dboxes(reader)

if __name__=='__main__ draw 3d boxes in rgb':
    reader = dataReader()
    draw_rgb_3dboxes(reader, pathlib.Path('outputs'))

if __name__=='__main__ draw 2d detection boxes in rgb':
    reader = dataReader()
    draw_rgb_2dboxes(reader, pathlib.Path('outputs'))

if __name__=='__main__ ':
    stereo_dir = pathlib.Path('mono_stereo/stereo')
    mono_dir = pathlib.Path('mono_stereo/mono')


    res_dir = stereo_dir
    #draw_lidar_3dboxes(dataReader())

    #res_dir = mono_dir
    draw_lidar_3dboxes(dataReader())

if __name__=='__main__ draw_3dboxes':
    stereo_dir = pathlib.Path('mono_stereo/stereo')
    mono_dir = pathlib.Path('mono_stereo/mono')


    res_dir = stereo_dir
    #draw_lidar_3dboxes(dataReader())

    #res_dir = mono_dir
    draw_rgb_3dboxes(dataReader(),pathlib.Path('outputs'))
