import numpy as np
import cv2

cam = cv2.VideoCapture(0)

bgs = cv2.createBackgroundSubtractorMOG2()

while(1):

    for i in range(5):
        cam.grab()
    ret, frame = cam.read()
    fgmask = bgs.apply(frame)
    mask = np.zeros((frame.shape[0], frame.shape[1], 3))
    mask[:,:,0]=fgmask//255
    mask[:,:,1]=fgmask//255
    mask[:,:,2]=fgmask//255
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
