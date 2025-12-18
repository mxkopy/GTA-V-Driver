import cupy, mmap, struct
from PIL import Image
import numpy as np

cudaArrayInfo = mmap.mmap(-1, 64 + 32, "CudaD3D11TextureArray1")
cudaArrayInfo.seek(0)
memHandle = cudaArrayInfo.read(64)
# print(len(cudaArrayInfo.readline()))
# bpp, pitch, height = struct.unpack("@3P", cudaArrayInfo.readline())
components, bpp, pitch, height = struct.unpack("@4P", cudaArrayInfo.readline())
print(components, bpp, pitch, height)


arrayPtr = cupy.cuda.runtime.ipcOpenMemHandle(memHandle)
membuffer = cupy.cuda.UnownedMemory(arrayPtr, pitch * height, owner=__name__, device_id=0)

array = cupy.ndarray(shape=(height, pitch // bpp), dtype=cupy.float32, memptr=cupy.cuda.MemoryPointer(membuffer, 0))
# array = cupy.asnumpy(array / array.max())
# array = array != 0

# print( np.sum(cupy.asnumpy(array) == 0) / (height * (pitch // bpp) ) )

# array = cupy.asnumpy(array) * 255
far = 100
C = 2
array = (np.pow(C*far+1,array)-1) / C

array = cupy.asnumpy(array) * 255
print(array.max())
Image.fromarray(array).show()