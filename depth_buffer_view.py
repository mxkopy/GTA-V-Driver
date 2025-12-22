import cv2
from environment import VideoState

vs = VideoState()

while True:
    keypress = cv2.waitKey(1)
    cv2.imshow("Test", vs.pop().squeeze().cpu().numpy())