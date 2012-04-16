float2 g_resolution;

sampler2D smp_occlusion;
sampler2D smp_position;
sampler2D smp_normal;
sampler2D smp_depth;
sampler2D smp_noise;


#define NUM_BLUR_SAMPLES 9
static const float blur_weights[NUM_BLUR_SAMPLES] =
{
	0.013519569015984728f,
	0.047662179108871855f,
	0.11723004402070096f,
	0.20116755999375591f,
	0.240841295721373f,
	0.20116755999375591f,
	0.11723004402070096f,
	0.047662179108871855f,
	0.013519569015984728f
};
static const float blur_indices[NUM_BLUR_SAMPLES] =
{
	-4,
	-3,
	-2,
	-1,
	0,
	+1,
	+2,
	+3,
	+4
};
static const float discard_threshold = 0.85;

float4 ssbn_blur(float2 baseTC, float2 direction) : COLOR0
{
	const float2 step = direction / g_resolution.xy;

	float2 sampleTC[NUM_BLUR_SAMPLES];
	float3 normal[NUM_BLUR_SAMPLES];	
	for(int i = 0; i < NUM_BLUR_SAMPLES; ++i)
	{
		sampleTC[i] = baseTC + blur_indices[i] * step;
		normal[i] = tex2D(smp_normal, sampleTC[i]) * 2 - 1;
	}	
	
	
	float total_weight = 0;	
	float4 res = 0;
	for(int j = 0; j < NUM_BLUR_SAMPLES; ++j)
	{
		float cond = (dot(normal[j], normal[4]) >= discard_threshold);	//0 when dot() < T, 1 when dot() >= T
		
		float weight = blur_weights[j] * cond;	//count only when dot() >= T
		
		res += tex2D(smp_occlusion, sampleTC[j]) * weight;
		total_weight += weight;
	}

	res /= total_weight;
	return res;

	return res;
}
