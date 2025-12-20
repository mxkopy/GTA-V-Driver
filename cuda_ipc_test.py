import cupy, mmap, struct
from PIL import Image
import numpy as np
import cv2

cudaArrayInfo = mmap.mmap(-1, 64 + 32, "CudaD3D11TextureArray1")
cudaArrayInfo.seek(0)
memHandle = cudaArrayInfo.read(64)
components, bpp, pitch, height = struct.unpack("@4P", cudaArrayInfo.readline())
print(components, bpp, pitch, height)
arrayPtr = cupy.cuda.runtime.ipcOpenMemHandle(memHandle)
membuffer = cupy.cuda.UnownedMemory(arrayPtr, pitch * height, owner=__name__, device_id=0)
array = cupy.ndarray(shape=(height, pitch // bpp), dtype=cupy.float32, memptr=cupy.cuda.MemoryPointer(membuffer, 0))

def linearize_depth(array, far=100000, C=0.001):
    array = (np.float_power(C*far+1,array)-1) / C
    array = np.clip(array, 0, 255)
    array = cupy.asnumpy(array) 
    array = array.astype(cupy.uint8)
    return array

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

# far = 100000
far = 10000
c = 0.001
while True:
    keypress = cv2.waitKey(1)
    if keypress != -1:
        print(far, c)
    if keypress == 119: # w
        far *= 10
    if keypress == 115: # s
        far /= 10
    if keypress == 97: # a
        c *= 10
    if keypress == 100: # d
        c /= 10
    cv2.imshow("Test", linearize_depth(array, far=far, C=c))
cv2.destroyAllWindows()

Image.fromarray(linearize_depth(array)).show()