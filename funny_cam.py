# https://github.com/kaustubh-sadekar/FunMirrors
import cv2
import numpy as np
from vcam import vcam, meshGen

cap = cv2.VideoCapture('videos/01.mp4')

ret, img = cap.read()
H, W = img.shape[:2]

c1 = vcam(H=H, W=W)
plane = meshGen(H, W)
plane_ori = meshGen(H, W)

plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
# plane.Z -= 10*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
# plane.Z -= 10*np.exp(-0.5*((plane.Y*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
# plane.Z += 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) + 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
# plane.Z -= 20*np.sin(2*np.pi*((plane.X-plane.W/4.0)/plane.W)) - 20*np.sin(2*np.pi*((plane.Y-plane.H/4.0)/plane.H))
# plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)

pts3d = plane.getPlane()
pts2d = c1.project(pts3d)
map_x, map_y = c1.getMaps(pts2d)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    output = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    cv2.imshow('img', img)
    cv2.imshow('output', output)
    if cv2.waitKey(1) == ord('q'):
        break
