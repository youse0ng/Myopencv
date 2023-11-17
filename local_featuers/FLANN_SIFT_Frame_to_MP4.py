# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import glob

img_shape=cv.imread('local_featuers/re_create/re_2.jpg').shape

out=cv.VideoWriter('output.mp4',cv.VideoWriter_fourcc(*'mp4v'),1,img_shape[0:2])
images= [f'local_featuers/re_create//re_{i}.jpg' for i in range(1,len(glob.glob('local_featuers/re_create/*.jpg'))+1)]

for image in images:
    img=cv.imread(image)
    out.write(img)
out.release()
print(cv.__version__())