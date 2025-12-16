// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"
#include "launchDebugger.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util
static bool debugged = false;

static std::ofstream logfile("dxinterop.log");

#ifndef NDEBUG
#define LOG(msg) logfile << msg << std::endl
#endif
#ifdef NDEBUG
#define LOG(msg)
#endif

static IDXGISwapChain* swapChain;
static DXGI_SWAP_CHAIN_DESC swapChainDesc;
static ID3D11Device* dev;
static ID3D11DeviceContext* ctx;

static void ERR(HRESULT err, const char* tag = NULL) {
    if (err != S_OK) {
        LOG(tag << " error " << err);
        throw std::system_error(err, std::system_category());
    }
}

static void CUERR(cudaError_t err, const char* tag = NULL) {
    if (err != cudaSuccess){
        LOG(tag << " cuda error " << cudaGetErrorString(err));
        throw std::system_error(err, std::system_category());
    };
}

static RECT winRect;
static void UPDATE_WINDOW_RECT(HWND window) {
    if (!GetWindowRect(window, &winRect)) throw std::system_error(S_SERDST, std::system_category());
}

static void getDeviceAndContextFromSwapChain(void* chain) {
    swapChain = static_cast<IDXGISwapChain*>(chain);
    ERR(swapChain->GetDesc(&swapChainDesc));
    ERR(swapChain->GetDevice(__uuidof(ID3D11Device), (LPVOID*)&dev));
    dev->GetImmediateContext(&ctx);
}

static int winH, winW;
static bool didSwapChainUpdate(HWND window) {
    UPDATE_WINDOW_RECT(window);
    bool winSizeChanged = (winH != winRect.bottom - winRect.top) || (winW != winRect.right - winRect.left);
    winH = winRect.bottom - winRect.top;
    winW = winRect.right - winRect.left;
    return winSizeChanged;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CUDA



// Depth buffer texture resources
static ID3D11DepthStencilState* depthStencilState;
static D3D11_DEPTH_STENCIL_DESC depthStencilStateDesc;
static ID3D11DepthStencilView* depthStencilView;
static CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
static ID3D11Resource* depthStencilViewResource;
static ID3D11Texture2D* depthStencilTexture;
static D3D11_TEXTURE2D_DESC depthStencilTextureDesc;
static ID3D11Texture2D* depthStencilBuffer;
static D3D11_TEXTURE2D_DESC depthStencilBufferDesc;


static ID3D11RenderTargetView* renderTargetView[8];
static D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
static ID3D11Resource* renderTargetResource;
static ID3D11Texture2D* renderTargetTexture;
static D3D11_TEXTURE2D_DESC renderTargetTextureDesc;

// CUDA IPC resources
static cudaGraphicsResource_t cudaResource;
static cudaArray_t cudaArray;
static void* cudaLinearArray;
static cudaIpcMemHandle_t cudaArrayIpcHandle;
static uintptr_t pitch;

void setupDepthBuffer(ID3D11Device* device) {

    depthStencilBufferDesc.Width = depthStencilTextureDesc.Width;
    depthStencilBufferDesc.Height = depthStencilTextureDesc.Height;
    depthStencilBufferDesc.MipLevels = 1;
    depthStencilBufferDesc.ArraySize = 1;
    //depthStencilBufferDesc.Format = depthStencilTextureDesc.Format;
    depthStencilBufferDesc.Format = DXGI_FORMAT_R32G32_FLOAT,
    depthStencilBufferDesc.SampleDesc.Count = 1;
    depthStencilBufferDesc.SampleDesc.Quality = 0;
    //depthStencilBufferDesc.Usage = D3D11_USAGE_STAGING;
    depthStencilBufferDesc.Usage = D3D11_USAGE_DEFAULT,
    //depthStencilBufferDesc.BindFlags = 0;
    depthStencilBufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    //depthStencilBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    depthStencilBufferDesc.CPUAccessFlags = 0;
    depthStencilBufferDesc.MiscFlags = 0;

    // Create depth stencil texture copy buffer
    ERR(device->CreateTexture2D(&depthStencilBufferDesc, NULL, &depthStencilBuffer), "device -> CreateTexture2D");

    // Register copy buffer to cuda IPC resource
    // TODO: see if you can return the cudaArray_t directly through the IPC handle without copying into the shared cuda memory
    CUERR(cudaGraphicsD3D11RegisterResource(&cudaResource, depthStencilBuffer, cudaGraphicsRegisterFlagsNone), "cudaGraphicsD3D11RegisterResource");

    // Allocate shared cuda memory 
    cudaFree(cudaLinearArray);
    cudaLinearArray = nullptr;
    CUERR(cudaMallocPitch(&cudaLinearArray, &pitch, 4 * depthStencilBufferDesc.Width, depthStencilBufferDesc.Height), "cudaMalloc");
}

static HANDLE cudaArrayMmapHandle;
static void** cudaArrayInfo;

static void MAP_CUINFO() {
    cudaArrayMmapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(cudaIpcMemHandle_t) + (2 * sizeof(uintptr_t)), L"cudaArrayInfo");
    cudaArrayInfo = (void**)MapViewOfFile(cudaArrayMmapHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
}

static void UNMAP_CUINFO() {
    UnmapViewOfFile(cudaArrayInfo);
    CloseHandle(cudaArrayMmapHandle);
}

static void writeCudaArrayInfo() {

    auto ipcInfo = (cudaIpcMemHandle_t*)cudaArrayInfo;
    auto arrayInfo = (uintptr_t*)cudaArrayInfo;
    auto offset = sizeof(cudaIpcMemHandle_t) / sizeof(uintptr_t);

    cudaIpcGetMemHandle(&cudaArrayIpcHandle, cudaLinearArray);
    ipcInfo[0] = cudaArrayIpcHandle;
    arrayInfo[offset + 0] = depthStencilBufferDesc.Height;
    arrayInfo[offset + 1] = depthStencilBufferDesc.Width;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Hooks

void DEBUG_RTV() {
    ID3D11Resource* rtr = nullptr;
    ID3D11Texture2D* rtt = nullptr;
    D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    D3D11_TEXTURE2D_DESC rttDesc = {};
    for (int i = 0; renderTargetView[i] != nullptr; i++) {
        renderTargetView[i]->GetDesc(&rtvDesc);
        renderTargetView[i]->GetResource(&rtr);
        ERR(rtr->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&rtt), "rtr->QueryInterface");
        rtt->GetDesc(&rttDesc);
        LOG("RT " << i);
        LOG(rttDesc.Width << ", " << rttDesc.Height);
        LOG("VFMT " << rtvDesc.Format << " TFMT " << rttDesc.Format);
        LOG("ViewDimension: " << rtvDesc.ViewDimension);
        LOG("BindFlags: " << rttDesc.BindFlags);
        LOG("ArraySize: " << rttDesc.ArraySize);
        LOG("Usage: " << rttDesc.Usage);
        LOG("");
    }
}

using ClearDepthStencilView_fn = void (__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView* pDepthStencilView, UINT clearFlags, FLOAT depth, UINT8 stencil);
static ClearDepthStencilView_fn ClearDepthStencilView = NULL;


static UINT tstencil;
static bool SECOND_DSV = true;

static D3D11_MAPPED_SUBRESOURCE sub;
static void* bytes = nullptr;
static size_t dsvWidth = 0;
static size_t dsvHeight = 0;

void ClearDepthStencilView_hook(ID3D11DeviceContext* self, ID3D11DepthStencilView* pDepthStencilView, UINT clearFlags, FLOAT depth, UINT8 stencil) {
    
    // The RTV is unbinded (unbound?) when the depth test is done, or something like that. I don't know for sure
    // In any case the real depth buffer is the one right before RTV gets NULL'd out
    // So we keep track of the last RTV, and if the current one is about to be NULL, the previous one corresponds w/ the true DTV
    //if (!debugged) launchDebugger();
    static ID3D11RenderTargetView* trueRTV;
    static ID3D11RenderTargetView* lastRTV;
    lastRTV = renderTargetView[0];

    self->OMGetDepthStencilState(&depthStencilState, &tstencil);
    ZeroMemory(renderTargetView, 8 * sizeof(ID3D11RenderTargetView*));
    //DebugBreak();
    self->OMGetRenderTargets(8, renderTargetView, &depthStencilView);
    DEBUG_RTV();
    depthStencilState->GetDesc(&depthStencilStateDesc);
    if (renderTargetView[0] == nullptr && lastRTV != nullptr && trueRTV == nullptr) trueRTV = lastRTV;

    if (renderTargetView[0] == trueRTV) {

        depthStencilView->GetDesc(&depthStencilViewDesc);
        depthStencilView->GetResource(&depthStencilViewResource);
        ERR(depthStencilViewResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&depthStencilTexture), "depthStencilViewResource -> QueryInterface");
        depthStencilTexture->GetDesc(&depthStencilTextureDesc);

        if (depthStencilBuffer == nullptr) setupDepthBuffer(dev);
        assert(depthStencilTexture != nullptr);
        assert(depthStencilBuffer != nullptr);
        self->CopyResource(depthStencilBuffer, depthStencilTexture);
        self->Flush();

#ifndef NDEBUG
        //D3D11_MAPPED_SUBRESOURCE sub = { 0 };
        //ERR(self->Map(depthStencilBuffer, 0, D3D11_MAP_READ, 0, &sub), "self -> Map");
        //size_t sum = 0;
        //for (int i = 0; i < sub.RowPitch; i++) sum += ((unsigned char*)(sub.pData))[i] != 0;
        //self->Unmap(depthStencilBuffer, 0);
        //LOG("nonzero/total: " << sum << "/" << sub.RowPitch);
#endif
        CUERR(cudaGraphicsMapResources(1, &cudaResource), "cudaGraphicsMapResource");
        CUERR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0), "cudaGraphicsSubResourceGetMappedArray");
        CUERR(cudaMemcpy2DFromArray(cudaLinearArray, pitch, cudaArray, 0, 0, 4 * depthStencilTextureDesc.Width, depthStencilTextureDesc.Height, cudaMemcpyDefault), "cudaMemcpy2D");
        CUERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        CUERR(cudaGraphicsUnmapResources(1, &cudaResource), "cudaGraphicsUnmapResources");
        writeCudaArrayInfo();
    }
    return ClearDepthStencilView(self, pDepthStencilView, clearFlags, depth, stencil);
}

static void hookClearDepthStencilView(){
    auto vtbl = (void**) *(void**) ctx;
    ClearDepthStencilView = (ClearDepthStencilView_fn)vtbl[53];
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(
        &(PVOID&)ClearDepthStencilView,
        (PBYTE*)&ClearDepthStencilView_hook
    );
    DetourTransactionCommit();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Main

// DX11 will launch an amortized version of ClearDepthStencilView once in a while
// and for whatever reason we do not want to hook that
// It happens only once in a while, so it seems we can reliably detect if we have
// the correct vtable entry by just checking that it hasn't changed 
static bool foundPrimaryVtableEntry = false;
static void* lastEntry = nullptr;
static bool hooked = false;

static void presentCallback(void* chain) {
    getDeviceAndContextFromSwapChain(chain);
    didSwapChainUpdate(swapChainDesc.OutputWindow);
    if (!hooked) {
        ID3D10Multithread* multiThread;
        ERR(ctx->QueryInterface(__uuidof(ID3D10Multithread), (void**)&multiThread));
        multiThread->SetMultithreadProtected(true);
        void** vtbl = (void**)*(void**)ctx;
        if (!foundPrimaryVtableEntry && (lastEntry == vtbl[53])) {
            foundPrimaryVtableEntry = true;
            hooked = true;
            hookClearDepthStencilView();
            //if (depthStencilBuffer == nullptr) setupDepthBuffer();
        }
        if (!foundPrimaryVtableEntry) lastEntry = vtbl[53];
    }
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    int deviceCount = 0;
    //hookClearDepthStencilView();
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        LOG("Attached!");
        //UPD::OpenDebugTerminal();
        //DebugBreak();
        //LOG("Attached!");
        //cudaGetDeviceCount(&deviceCount);
        //LOG("Device count:");
        //LOG(deviceCount);
        MAP_CUINFO();
        presentCallbackRegister(presentCallback);
        break;
    case DLL_PROCESS_DETACH:
        presentCallbackUnregister(presentCallback);
        cudaGraphicsUnregisterResource(cudaResource);
        UNMAP_CUINFO();
        break;
    }
    return TRUE;
}

