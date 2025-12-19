Texture2D<float3> InputTexture : register(t);
RWTexture2D<float> OutputTexture : register(u);

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    OutputTexture[DTid.xy].r = InputTexture[DTid.xy].r;
}