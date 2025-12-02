// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"


static void* SwapChainPtr = NULL;

__declspec(dllexport) void* getSwapChainPtr() {
    return SwapChainPtr;
}

void presentCallback(void* swapchain) {
    SwapChainPtr = swapchain;
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        presentCallbackRegister(presentCallback);
        break;
    case DLL_PROCESS_DETACH:
        presentCallbackUnregister(presentCallback);
        break;
    }
    return TRUE;
}

