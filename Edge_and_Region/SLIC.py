import skimage
import numpy as np
import cv2 as cv

img = cv.imread("Lena.jpg")
cv.imshow("coffee image",cv.cvtColor(img,cv.COLOR_BGR2GRAY))

slic1=skimage.segmentation.slic(img,compactness=20,n_segments=600)
sp_img1=skimage.segmentation.mark_boundaries(img,slic1)
sp_img1=np.uint8(sp_img1 * 255.0)

slic2=skimage.segmentation.slic(img,compactness=40,n_segments=600)
sp_img2=skimage.segmentation.mark_boundaries(img,slic2)
sp_img2=np.uint8(sp_img2 * 255.0)

cv.imshow("Super pixels (compact 20)", cv.cvtColor(sp_img1,cv.COLOR_RGB2BGR))
cv.imshow("Super pixels (compact 40)", cv.cvtColor(sp_img2,cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()