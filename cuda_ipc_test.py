import cupy, mmap, struct
from PIL import Image

cudaArrayInfo = mmap.mmap(-1, 64 + 16, "cudaArrayInfo")
cudaArrayInfo.seek(0)
memHandle = cudaArrayInfo.read(64)
# print(len(cudaArrayInfo.readline()))
height, width = struct.unpack("@2P", cudaArrayInfo.readline())
arrayPtr = cupy.cuda.runtime.ipcOpenMemHandle(memHandle)
membuffer = cupy.cuda.UnownedMemory(arrayPtr, 4 * height * width, owner=None, device_id=0)
array = cupy.ndarray(shape=(height, width, 4), dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(membuffer, 0))
print(array.sum())
Image.fromarray(cupy.asnumpy(array)).show()