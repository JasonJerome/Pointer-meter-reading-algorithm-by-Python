import cv2 as cv
import numpy as np

m=[0,0,1,0,0,1,1,1]
A=np.array(m).reshape(4,2).astype(np.float32)
m=cv.minAreaRect(A)
print(m)
print(cv.__version__)