#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
#include <system_error>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <dxgi.h>
#include <d3d11.h>
#include <d3d11shader.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <driver_types.h>
#include <wrl/client.h>
#include <assert.h>
#include "detours.h"
#include "UltimateProxyDLL.h"
#include "main.h"

/*extern IDXGISwapChain* swapChain;
extern ID3D11Device* dev;
extern ID3D11DeviceContext* ctx;
extern ID3D11Texture2D* depthStencilBuffer;
extern ID3D11DepthStencilView* depthStencilView;
extern ID3D11ShaderResourceView* shaderResourceView;
extern cudaGraphicsResource* cudaResource;
extern HANDLE cuDevPtrMmap;
extern HANDLE cuDevPtrSizeMmap;
extern void* cuDevPtrMmapView;
extern void* cuDevPtrSizeMmapView;
extern void* cuDevPtr;
extern size_t cuDevPtrSize;*/
