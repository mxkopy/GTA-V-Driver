import cupy, mmap, struct
from PIL import Image
import numpy as np
import cv2
from environment import VideoState
from ipc import Channel

# mm = mmap.mmap(-1, 64, "CudaArray1")
# memHandle = mm.read(64)
# components, bpp, pitch, height = struct.unpack("@4P", mmap.mmap(-1, 32, "CudaArray1Info"))
# print(components, bpp, pitch, height)
# arrayPtr = cupy.cuda.runtime.ipcOpenMemHandle(memHandle)
# membuffer = cupy.cuda.UnownedMemory(arrayPtr, pitch * height, owner=__name__, device_id=0)
# array = cupy.ndarray(shape=(height, pitch // bpp), dtype=cupy.float32, memptr=cupy.cuda.MemoryPointer(membuffer, 0))

# def linearize_depth(array, far=100000, C=0.001):
#     array = (np.float_power(C*far+1,array)-1) / C
#     array = np.clip(array, 0, 255)
#     array = cupy.asnumpy(array) 
#     array = array.astype(cupy.uint8)
#     return array

vs = VideoState()

# Image.fromarray(vs.pop().squeeze().cpu().numpy()).show()
# exit()
# cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

while True:
    keypress = cv2.waitKey(1)
    cv2.imshow("Test", vs.pop().squeeze().cpu().numpy())
cv2.destroyAllWindows()