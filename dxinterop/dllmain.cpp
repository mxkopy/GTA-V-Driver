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

// Depth info buffer, gets copied to cuda memory
// Unfortunately there doesn't seem to be a straightforward way to read the depth stencil texture directly into cuda memory
// So we have to have a sort of 'staging' buffer that 

void CreateDepthTextureBuffer
(
    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    int BindFlags,
    DXGI_FORMAT Format,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
) {
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    GetTextureFromView(DepthStencilView, &DepthStencilTextureDesc);

    D3D11_TEXTURE2D_DESC BufferDesc;
    BufferDesc.Width = DepthStencilTextureDesc.Width;
    BufferDesc.Height = DepthStencilTextureDesc.Height;
    BufferDesc.MipLevels = 1;
    BufferDesc.ArraySize = 1;
    BufferDesc.Format = Format;
    BufferDesc.SampleDesc.Count = 1;
    BufferDesc.SampleDesc.Quality = 0;
    BufferDesc.Usage = D3D11_USAGE_DEFAULT;
    BufferDesc.BindFlags = BindFlags;
    BufferDesc.CPUAccessFlags = 0;
    BufferDesc.MiscFlags = 0;

    ERR(Device->CreateTexture2D(&BufferDesc, NULL, &DepthTextureBuffer));
}

void CreateDepthTextureBuffer
(
    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    int BindFlags,
    DXGI_FORMAT Format,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer
) {
    ComPtr<ID3D11Device> Device;
    DeviceContext->GetDevice(&Device);
    CreateDepthTextureBuffer(Device, DepthStencilView, BindFlags, Format, DepthTextureBuffer);
}

// For reading from depth stencil texture
// Should convert formats, i.e. DXGI_FORMAT_R32G8X24_TYPELESS -> DXGI_FORMAT_R32X8X24_TYPELESS 

void CreateDepthStencilShaderResourceView
(
    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11Texture2D> DepthStencilTexture,
    ComPtr<ID3D11ShaderResourceView>& ShaderResourceView
) {
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    DepthStencilTexture->GetDesc(&DepthStencilTextureDesc);

    D3D11_SHADER_RESOURCE_VIEW_DESC ShaderResourceViewDesc;
    ShaderResourceViewDesc.Format = GetDepthFormatFromDepthStencilFormat(DepthStencilTextureDesc.Format);
    ShaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    ShaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
    ShaderResourceViewDesc.Texture2D.MipLevels = -1;

    ERR(Device->CreateShaderResourceView(DepthStencilTexture.Get(), &ShaderResourceViewDesc, ShaderResourceView.GetAddressOf()));
}

// For writing to depth texture 
// Depth texture should have a depth-only format (e.g. DXGI_FORMAT_R32_FLOAT), so no need to convert

void CreateDepthStencilUnorderedAccessView
(
    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11Texture2D> DepthStencilBuffer,
    ComPtr<ID3D11UnorderedAccessView>& UnorderedAccessView
) {
    D3D11_TEXTURE2D_DESC DepthStencilBufferDesc;
    DepthStencilBuffer->GetDesc(&DepthStencilBufferDesc);

    D3D11_UNORDERED_ACCESS_VIEW_DESC UnorderedAccessViewDesc;
    UnorderedAccessViewDesc.Format = DepthStencilBufferDesc.Format;
    UnorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    UnorderedAccessViewDesc.Texture2D.MipSlice = 0;

    ERR(Device->CreateUnorderedAccessView(DepthStencilBuffer.Get(), &UnorderedAccessViewDesc, UnorderedAccessView.GetAddressOf()));
}


using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;

void CreateComputeShader
(
    ComPtr<ID3D11Device> Device, 
    ComPtr<ID3D11ComputeShader>& ComputeShader
) {

    const char Shader[] =
    R"(
        Texture2D<float3> InputTexture : register(t);
        RWTexture2D<float> OutputTexture : register(u);

        [numthreads(32, 32, 1)]
        void main(uint3 DTid : SV_DispatchThreadID)
        {
            OutputTexture[DTid.xy].r = InputTexture[DTid.xy].r;
        }
    )";

    ComPtr<ID3DBlob> ShaderBlob;
    ComPtr<ID3DBlob> ErrorBlob;

    HRESULT HR = D3DCompile
    (
        Shader,                             // SrcData
        sizeof(Shader),                     // SrcDataSize
        NULL,                               // SourceName
        NULL,                               // Defines
        D3D_COMPILE_STANDARD_FILE_INCLUDE,  // Include 
        "main",                             // EntryPoint
        "cs_5_0",                           // Target
        NULL,                               // Flags1
        NULL,                               // Flags2
        ShaderBlob.GetAddressOf(),          // Code
        ErrorBlob.GetAddressOf()            // ErrorMsgs
    );

    if (HR != S_OK) logfile << std::string((char*)ErrorBlob->GetBufferPointer(), ErrorBlob->GetBufferSize());
    ERR(HR);
    ERR(Device->CreateComputeShader(ShaderBlob->GetBufferPointer(), ShaderBlob->GetBufferSize(), NULL, ComputeShader.GetAddressOf()));
}

void RunComputeShader
(
    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11ComputeShader> ComputeShader,
    ComPtr<ID3D11ShaderResourceView> ShaderResourceView,
    ComPtr<ID3D11UnorderedAccessView> UnorderedAccessView,
    UINT X,
    UINT Y
) {

    DeviceContext->CSSetShader(ComputeShader.Get(), NULL, NULL);
    DeviceContext->CSSetShaderResources(0, 1, ShaderResourceView.GetAddressOf());
    DeviceContext->CSSetUnorderedAccessViews(0, 1, UnorderedAccessView.GetAddressOf(), NULL);

    DeviceContext->Dispatch(X, Y, 1);
    DeviceContext->Flush();

    DeviceContext->CSSetShader(nullptr, nullptr, 0);

    ID3D11UnorderedAccessView* NullUAV[1] = { nullptr };
    ID3D11ShaderResourceView* NullSRV[1] = { nullptr };

    DeviceContext->CSSetUnorderedAccessViews(0, 1, NullUAV, nullptr);
    DeviceContext->CSSetShaderResources(0, 1, NullSRV);

}


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
    static ComPtr<ID3D11ComputeShader> ComputeShader;

    static ComPtr<ID3D11DepthStencilView> DepthStencilView;
    static ComPtr<ID3D11Texture2D> DepthStencilTexture;
    static CD3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    static ComPtr<ID3D11ShaderResourceView> DepthStencilShaderResourceView;
    static ComPtr<ID3D11Texture2D> DepthStencilBuffer;
    static ComPtr<ID3D11UnorderedAccessView> DepthStencilUnorderedAccessView;

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

    if (pRenderTargetView == pTrueRTV && CudaArray.Memory == nullptr)
    {
        GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);
        DEBUG_TEXTURE2D(DepthStencilTexture, "DepthStencilTexture");
        CreateDepthTextureBuffer(DeviceContext, DepthStencilView, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, DXGI_FORMAT_R32_FLOAT, DepthStencilBuffer);
        CreateDepthStencilShaderResourceView(Device, DepthStencilTexture, DepthStencilShaderResourceView);
        CreateDepthStencilUnorderedAccessView(Device, DepthStencilBuffer, DepthStencilUnorderedAccessView);
        CreateComputeShader(Device, ComputeShader);
        CudaArray = CudaD3D11TextureArray(DepthStencilBuffer, 1);
    }


    if (CudaArray.Memory != nullptr)
    {
        
        RunComputeShader
        (
            DeviceContext, 
            ComputeShader, 
            DepthStencilShaderResourceView, 
            DepthStencilUnorderedAccessView, 
            (DepthStencilTextureDesc.Width + 32) / 32,
            (DepthStencilTextureDesc.Height + 32) / 32
        );
        CUERR(cudaDeviceSynchronize());
        CudaArray.Update();
    }

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

static void presentCallback(void* chain) {

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

BOOL APIENTRY DllMain
(
    HMODULE hModule,                   
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
) {
    int DeviceCount = 0;
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        cudaGetDeviceCount(&DeviceCount);
        LOG("Attached! Cuda Device Count: " << DeviceCount << std::endl);
        presentCallbackRegister(presentCallback);
        break;
    case DLL_PROCESS_DETACH:
        presentCallbackUnregister(presentCallback);
        CudaD3D11TextureArray::DestroyAll();
        break;
    }
    return TRUE;
}

