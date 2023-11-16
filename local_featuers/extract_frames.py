import cv2 as cv
import os 

vidcap = cv.VideoCapture('local_featuers\soccer.mp4')
success,image = vidcap.read()
count=0

while success:
  success,image = vidcap.read()
  cv.imwrite(f'{count}.jpg',image)
  print("saved image %d.jpg" % count)
  
  if cv.waitKey(10) == 27:                    
      break
  count += 1