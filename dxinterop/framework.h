#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include "main.h"
#include <d3d11.h>

extern void* SwapChainPtr;

__declspec(dllexport) void* getSwapChainPtr();