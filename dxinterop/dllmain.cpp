// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"
#include "launchDebugger.h"

using Microsoft::WRL::ComPtr;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util
static bool debugged = false;
static std::ofstream logfile("dxinterop.log");

#ifndef NDEBUG
#define LOG(msg) logfile << msg << std::endl
#else
#define LOG(msg)
#endif

static ComPtr<IDXGISwapChain> SwapChain;
static DXGI_SWAP_CHAIN_DESC SwapChainDesc;
static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> DeviceContext;
static void** DeviceContextVirtualTable;

HRESULT _ERR;
cudaError_t _CUERR;

#define ERR(CALL)\
_ERR = CALL;\
if(_ERR != S_OK){\
    LOG(#CALL << " returned error: " << _ERR);\
    throw std::system_error(_ERR, std::system_category());\
}

#define CUERR(CALL)\
_CUERR = CALL;\
if(_CUERR != cudaSuccess){\
    LOG(#CALL << " returned error: " << cudaGetErrorString(_CUERR));\
    throw std::system_error(_CUERR, std::system_category());\
}


static RECT winRect;
static void UPDATE_WINDOW_RECT(HWND window) {
    if (!GetWindowRect(window, &winRect)) throw std::system_error(S_SERDST, std::system_category());
}

static void GetDeviceAndContextFromSwapChain(void* chain) {
    SwapChain = (IDXGISwapChain*) chain;
    ERR(SwapChain->GetDesc(&SwapChainDesc));
    ERR(SwapChain->GetDevice(__uuidof(ID3D11Device), &Device));
    Device->GetImmediateContext(&DeviceContext);
    DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
}

static bool DidSwapChainUpdate(HWND window) {
    static int winH, winW;
    UPDATE_WINDOW_RECT(window);
    bool winSizeChanged = (winH != winRect.bottom - winRect.top) || (winW != winRect.right - winRect.left);
    winH = winRect.bottom - winRect.top;
    winW = winRect.right - winRect.left;
    return winSizeChanged;
}

void DEBUG_TEXTURE2D(ComPtr<ID3D11Texture2D> Texture, const char* Name = "DEBUG") 
{
    D3D11_TEXTURE2D_DESC TextureDesc;
    Texture->GetDesc(&TextureDesc);
    LOG(Name);
    LOG("Shape          " << TextureDesc.Width << ", " << TextureDesc.Height);
    LOG("Format         " << TextureDesc.Format);
    LOG("BindFlags:     " << TextureDesc.BindFlags);
    LOG("MiscFlags      " << TextureDesc.MiscFlags);
    LOG("SampleCount:   " << TextureDesc.SampleDesc.Count);
    LOG("SampleQuality: " << TextureDesc.SampleDesc.Quality);
    LOG("MipLevels:     " << TextureDesc.MipLevels);
    LOG("ArraySize:     " << TextureDesc.ArraySize);
    LOG("Usage:         " << TextureDesc.Usage);
    LOG("");
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Compute Shader


DXGI_FORMAT GetDepthFormatFromDepthStencilFormat(DXGI_FORMAT Format)
{
    switch (Format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

template<typename T>
concept IsD3D11View = std::is_base_of_v<ID3D11View, T>;

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, ComPtr<ID3D11Texture2D>& Texture)
{
    ComPtr<ID3D11Resource> Resource;
    View->GetResource(&Resource);
    ERR(Resource.As(&Texture));
}

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, ComPtr<ID3D11Texture2D>& Texture, D3D11_TEXTURE2D_DESC* TextureDesc)
{
    GetTextureFromView(View, Texture);
    Texture->GetDesc(TextureDesc);
}

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, D3D11_TEXTURE2D_DESC* TextureDesc)
{
    ComPtr<ID3D11Texture2D> Texture;
    GetTextureFromView(View, Texture);
    Texture->GetDesc(TextureDesc);
}

void CreateDepthStencilShaderResourceView
(
    ComPtr<ID3D11Device> Device, 
    ComPtr<ID3D11DepthStencilView> DepthStencilView, 
    ComPtr<ID3D11ShaderResourceView>& ShaderResourceView
) {
    ComPtr<ID3D11Texture2D> DepthStencilTexture;
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);

    D3D11_SHADER_RESOURCE_VIEW_DESC ShaderResourceViewDesc;  
    ShaderResourceViewDesc.Format = GetDepthFormatFromDepthStencilFormat(DepthStencilTextureDesc.Format);
    ShaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    ShaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
    ShaderResourceViewDesc.Texture2D.MipLevels = -1;

    Device->CreateShaderResourceView(DepthStencilTexture.Get(), &ShaderResourceViewDesc, ShaderResourceView.GetAddressOf());
}


void CreateDepthTextureStagingBuffer
(
    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
)
{
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    GetTextureFromView(DepthStencilView, &DepthStencilTextureDesc);

    D3D11_TEXTURE2D_DESC BufferDesc;
    BufferDesc.Width = DepthStencilTextureDesc.Width;
    BufferDesc.Height = DepthStencilTextureDesc.Height;
    BufferDesc.MipLevels = 1;
    BufferDesc.ArraySize = 1;
    BufferDesc.Format = DXGI_FORMAT_R32G8X24_TYPELESS,
    BufferDesc.SampleDesc.Count = 1;
    BufferDesc.SampleDesc.Quality = 0;
    BufferDesc.Usage = D3D11_USAGE_STAGING,
    BufferDesc.BindFlags = 0;
    BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    BufferDesc.MiscFlags = 0;

    ERR(Device->CreateTexture2D(&BufferDesc, NULL, &DepthTextureBuffer));
}

void CreateDepthTextureStagingBuffer
(
    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
)
{
    ComPtr<ID3D11Device> Device;
    DeviceContext->GetDevice(&Device);
    CreateDepthTextureStagingBuffer(Device, DepthStencilView, DepthTextureBuffer);
}


void CreateDepthTextureBuffer
(
    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
)
{
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    GetTextureFromView(DepthStencilView, &DepthStencilTextureDesc);


    D3D11_TEXTURE2D_DESC BufferDesc;
    BufferDesc.Width = DepthStencilTextureDesc.Width;
    BufferDesc.Height = DepthStencilTextureDesc.Height;
    BufferDesc.MipLevels = 1;
    BufferDesc.ArraySize = 1;
    BufferDesc.Format = DXGI_FORMAT_R32_FLOAT,
    BufferDesc.SampleDesc.Count = 1;
    BufferDesc.SampleDesc.Quality = 0;
    BufferDesc.Usage = D3D11_USAGE_DYNAMIC,
    BufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    BufferDesc.MiscFlags = 0;

    //D3D11_TEXTURE2D_DESC BufferDesc;
    //BufferDesc.Width = DepthStencilTextureDesc.Width;
    //BufferDesc.Height = DepthStencilTextureDesc.Height;
    //BufferDesc.MipLevels = 1;
    //BufferDesc.ArraySize = 1;
    //BufferDesc.Format = DXGI_FORMAT_R32_FLOAT,
    //BufferDesc.SampleDesc.Count = 1;
    //BufferDesc.SampleDesc.Quality = 0;
    //BufferDesc.Usage = D3D11_USAGE_DEFAULT,
    //BufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    //BufferDesc.CPUAccessFlags = 0;
    //BufferDesc.MiscFlags = 0;

    ERR(Device->CreateTexture2D(&BufferDesc, NULL, &DepthTextureBuffer));
}

void CreateDepthTextureBuffer
(
    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
)
{
    ComPtr<ID3D11Device> Device;
    DeviceContext->GetDevice(&Device);
    CreateDepthTextureBuffer(Device, DepthStencilView, DepthTextureBuffer);
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CUDA


cudaChannelFormatDesc CudaChannelFormatFromDXGIFormat(DXGI_FORMAT Format)
{
    switch (Format)
    {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return { 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return { 32, 32, 0, 0, cudaChannelFormatKindNone };
    case DXGI_FORMAT_R32G32_FLOAT:
        return { 32, 32, 0, 0, cudaChannelFormatKindFloat };
    case DXGI_FORMAT_R32_FLOAT:
        return { 32, 0, 0, 0, cudaChannelFormatKindFloat };
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

UINT GetBytesPerPixelFromDXGIFormat(DXGI_FORMAT Format)
{
    cudaChannelFormatDesc ChannelFormat = CudaChannelFormatFromDXGIFormat(Format);
    return (ChannelFormat.x + ChannelFormat.y + ChannelFormat.z + ChannelFormat.w) / 8;
}

struct CudaD3D11TextureArray
{
    inline static CudaD3D11TextureArray* Instances[10] = { 0 };

    void*                  Memory = nullptr;
    cudaChannelFormatDesc  ChannelFormat = {};
    uint64_t               BPP = {};
    uint64_t               Pitch = {};
    cudaExtent             Extent = {};

    HANDLE                 InfoMmapHandle = {};
    void*                  InfoMmap = nullptr;

    cudaGraphicsResource_t GraphicsResource = {};
    cudaIpcMemHandle_t     IPCMemHandle = {};

    CudaD3D11TextureArray() = default;

    CudaD3D11TextureArray(ComPtr<ID3D11Texture2D>& Texture, int ID = 0)
    {
        DEBUG_TEXTURE2D(Texture, ("ID " + std::to_string(ID)).c_str());

        Instances[ID] = this;
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        BPP = GetBytesPerPixelFromDXGIFormat(TextureDesc.Format);
        CUERR(cudaMallocPitch(&Memory, &Pitch, BPP * TextureDesc.Width, TextureDesc.Height));
        CUERR(cudaIpcGetMemHandle(&IPCMemHandle, Memory));
        ChannelFormat = CudaChannelFormatFromDXGIFormat(TextureDesc.Format);
        Extent.width = TextureDesc.Width;
        Extent.height = TextureDesc.Height;
        Extent.depth = 1;
        CUERR(cudaGraphicsD3D11RegisterResource(&GraphicsResource, Texture.Get(), cudaGraphicsRegisterFlagsNone));
        InfoMmapHandle = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            sizeof(cudaIpcMemHandle_t) + (4 * sizeof(uint64_t)),
            (L"CudaD3D11TextureArray" + std::to_wstring(ID)).c_str()
        );
        InfoMmap = MapViewOfFile(InfoMmapHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    }

    void WriteInfo()
    {
        CUERR(cudaIpcGetMemHandle(&IPCMemHandle, Memory));
        ((cudaIpcMemHandle_t*)InfoMmap)[0] = IPCMemHandle;
        auto UIntCudaArrayInfoMmap = (uint64_t*)InfoMmap + (sizeof(cudaIpcMemHandle_t) / sizeof(uint64_t));
        UIntCudaArrayInfoMmap[0] = (ChannelFormat.x > 0) + (ChannelFormat.y > 0) + (ChannelFormat.z > 0) + (ChannelFormat.w > 0);
        UIntCudaArrayInfoMmap[1] = BPP;
        UIntCudaArrayInfoMmap[2] = Pitch;
        UIntCudaArrayInfoMmap[3] = Extent.height;
    }

    void CopyFrom(cudaArray_t CudaArray)
    {
        CUERR(cudaMemcpy2DFromArray(Memory, Pitch, CudaArray, 0, 0, BPP * Extent.width, Extent.height, cudaMemcpyDefault));
    }

    void Update()
    {
        cudaArray_t MappedArray;
        CUERR(cudaGraphicsMapResources(1, &GraphicsResource));
        CUERR(cudaGraphicsSubResourceGetMappedArray(&MappedArray, GraphicsResource, 0, 0));
        CopyFrom(MappedArray);
        CUERR(cudaGraphicsUnmapResources(1, &GraphicsResource));
        WriteInfo();
    }

    void Destroy()
    {
        UnmapViewOfFile(InfoMmap);
        CloseHandle(InfoMmapHandle);
        cudaGraphicsUnregisterResource(GraphicsResource);
        cudaFree(Memory);
    }

    static void DestroyAll()
    {
        for (auto Array : CudaD3D11TextureArray::Instances) Array->Destroy();
    }

};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Hooks



using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;


void ClearDepthStencilViewHook
(
    ID3D11DeviceContext* pDeviceContext, 
    ID3D11DepthStencilView* pDepthStencilView, 
    UINT clearFlags, 
    FLOAT depth, 
    UINT8 stencil
) {

    ComPtr<ID3D11DeviceContext> DeviceContext = pDeviceContext;

    static CudaD3D11TextureArray CudaArray;

    static ComPtr<ID3D11DepthStencilView> DepthStencilView;
    static ComPtr<ID3D11Texture2D> DepthStencilTexture;
    static ComPtr<ID3D11Texture2D> DepthStencilBuffer;

    // The RTV is unbinded (unbound?) when the depth test is done, or something like that. I don't know for sure
    // In any case the real depth buffer is the one right before RTV gets NULL'd out
    // So we keep track of the last RTV, and if the current one is about to be NULL, the previous one corresponds w/ the true DTV

    static ID3D11RenderTargetView* pRenderTargetView;
    static ID3D11RenderTargetView* pLastRenderTargetView;
    static ID3D11RenderTargetView* pTrueRTV;

    pLastRenderTargetView = pRenderTargetView;
    DeviceContext->OMGetRenderTargets(1, &pRenderTargetView, &DepthStencilView);

    if (pRenderTargetView == nullptr && pLastRenderTargetView != nullptr && pTrueRTV == nullptr)
    {
        pTrueRTV = pLastRenderTargetView;
    }

    static std::vector<float> Depth;
    static ComPtr<ID3D11Texture2D> DepthStencilStagingBuffer;

    if (pRenderTargetView == pTrueRTV && CudaArray.Memory == nullptr)
    {
        GetTextureFromView(DepthStencilView, DepthStencilTexture);
        DEBUG_TEXTURE2D(DepthStencilTexture, "DepthStencilTexture");
        CreateDepthTextureBuffer(DeviceContext, DepthStencilView, DepthStencilBuffer);
        CudaArray = CudaD3D11TextureArray(DepthStencilBuffer, 1);

        CreateDepthTextureStagingBuffer(DeviceContext, DepthStencilView, DepthStencilStagingBuffer);
        Depth.resize(CudaArray.Extent.width * CudaArray.Extent.height);
    }


    if (CudaArray.Memory != nullptr)
    {
        //DeviceContext->CopyResource(DepthStencilBuffer.Get(), DepthStencilTexture.Get());
    
        // TODO: Replace this all w/ a compute shader
        DeviceContext->CopyResource(DepthStencilStagingBuffer.Get(), DepthStencilTexture.Get());

        D3D11_MAPPED_SUBRESOURCE MappedSubresource1 = { 0 };
        D3D11_MAPPED_SUBRESOURCE MappedSubresource2 = { 0 };
        ERR(DeviceContext->Map(DepthStencilStagingBuffer.Get(), 0, D3D11_MAP_READ, 0, &MappedSubresource1));
        ERR(DeviceContext->Map(DepthStencilBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubresource2));
        float* MappedData1 = (float*) MappedSubresource1.pData;
        float* MappedData2 = (float*) MappedSubresource2.pData;

        static bool logged;
        if (!logged)
        {
            LOG("RP1 " << MappedSubresource1.RowPitch);
            LOG("DP1 " << MappedSubresource1.DepthPitch);
            LOG("");
            LOG("RP2 " << MappedSubresource2.RowPitch);
            LOG("DP2 " << MappedSubresource2.DepthPitch);
            logged = true;
        }
       
        for (int i = 0; i < MappedSubresource2.DepthPitch / sizeof(float); i++) MappedData2[i] = MappedData1[i*2];
        
        DeviceContext->Unmap(DepthStencilBuffer.Get(), 0);
        DeviceContext->Unmap(DepthStencilStagingBuffer.Get(), 0);
        DeviceContext->Flush();

        CUERR(cudaDeviceSynchronize());

        CudaArray.Update();
    }
    //DeviceContext.Detach();

    return ClearDepthStencilView(pDeviceContext, pDepthStencilView, clearFlags, depth, stencil);
}

static void HookClearDepthStencilView()
{
    ClearDepthStencilView = (ClearDepthStencilViewFunction)DeviceContextVirtualTable[53];
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach
    (
        &(PVOID&)ClearDepthStencilView,
        (PBYTE*)&ClearDepthStencilViewHook
    );
    DetourTransactionCommit();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Main

static cudaGraphicsResource_t CudaRenderTargetResource;

static void presentCallback(void* chain) {
    //LaunchDebugger();
    //DebugBreak();

    static ComPtr<ID3D10Multithread> MultithreadContext;

    GetDeviceAndContextFromSwapChain(chain);
    DidSwapChainUpdate(SwapChainDesc.OutputWindow);

    // DX11 will launch an amortized version of ClearDepthStencilView once in a while
    // and for whatever reason we do not want to hook that
    // It happens only occasionally, so it seems we can reliably detect if we have
    // the correct vtable entry by just checking that it hasn't changed 

    static void* LastVTEntry;
    static bool FoundPrimaryVTEntry;
    static bool Hooked;

    if (!Hooked)
    {
        DeviceContext.As(&MultithreadContext);
        MultithreadContext->SetMultithreadProtected(true);
        DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
        if (!FoundPrimaryVTEntry && LastVTEntry == DeviceContextVirtualTable[53])
        {
            HookClearDepthStencilView();
            Hooked = true;
            FoundPrimaryVTEntry = true;
        }
    }

    LastVTEntry = DeviceContextVirtualTable[53];


    // RTV hook related stuff (i.e. the getting displayed pixels)
    static CudaD3D11TextureArray RenderTargetArray;
    static ComPtr<ID3D11RenderTargetView> RenderTargetView;
    static ComPtr<ID3D11Resource> RenderTargetResource;
    static ComPtr<ID3D11Texture2D> RenderTargetTexture;

    DeviceContext->OMGetRenderTargets(1, RenderTargetView.GetAddressOf(), NULL);

    if (RenderTargetTexture == nullptr)
    {
        GetTextureFromView(RenderTargetView, RenderTargetTexture);
        RenderTargetArray = CudaD3D11TextureArray(RenderTargetTexture, 0);
    }
    else
    {
        RenderTargetArray.Update();
    }
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    int deviceCount = 0;
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        LOG("Attached!");
        //cudaGetDeviceCount(&deviceCount);
        //LOG("Device count:");
        //LOG(deviceCount);
        presentCallbackRegister(presentCallback);
        break;
    case DLL_PROCESS_DETACH:
        presentCallbackUnregister(presentCallback);
        CudaD3D11TextureArray::DestroyAll();
        break;
    }
    return TRUE;
}

