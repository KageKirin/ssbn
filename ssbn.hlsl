float g_occlusion_radius;
float g_occlusion_max_distance;

float4x4 g_mat_view_inv;
float2 g_resolution;

float fFarClipPlane;
float fNearClipPlane;
float4x4 matViewProjection;

const int OCCLUSION_SAMPLE_COUNT = 32;

sampler2D smp_occlusion;
sampler2D smp_position;
sampler2D smp_normal;
sampler2D smp_depth;
sampler2D smp_noise;

#define KERNEL_SAMPLES_32	\
	{	half3(-0.556641,		-0.037109,		-0.654297),	\
		half3(0.173828,			0.111328,		0.064453),	\
		half3(0.001953,			0.082031,		-0.060547),	\
		half3(0.220703,			-0.359375,		-0.062500),	\
		half3(0.242188,			0.126953,		-0.250000),	\
		half3(0.070313,			-0.025391,		0.148438),	\
		half3(-0.078125,		0.013672,		-0.314453),	\
		half3(0.117188,			-0.140625,		-0.199219),	\
		half3(-0.251953,		-0.558594,		0.082031),	\
		half3(0.308594,			0.193359,		0.324219),	\
		half3(0.173828,			-0.140625,		0.031250),	\
		half3(0.179688,			-0.044922,		0.046875),	\
		half3(-0.146484,		-0.201172,		-0.029297),	\
		half3(-0.300781,		0.234375,		0.539063),	\
		half3(0.228516,			0.154297,		-0.119141),	\
		half3(-0.119141,		-0.003906,		-0.066406),	\
		half3(-0.218750,		0.214844,		-0.250000),	\
		half3(0.113281,			-0.091797,		0.212891),	\
		half3(0.105469,			-0.039063,		-0.019531),	\
		half3(-0.705078,		-0.060547,		0.023438),	\
		half3(0.021484,			0.326172,		0.115234),	\
		half3(0.353516,			0.208984,		-0.294922),	\
		half3(-0.029297,		-0.259766,		0.089844),	\
		half3(-0.240234,		0.146484,		-0.068359),	\
		half3(-0.296875,		0.410156,		-0.291016),	\
		half3(0.078125,			0.113281,		-0.126953),	\
		half3(-0.152344,		-0.019531,		0.142578),	\
		half3(-0.214844,		-0.175781,		0.191406),	\
		half3(0.134766,			0.414063,		-0.707031),	\
		half3(0.291016,			-0.833984,		-0.183594),	\
		half3(-0.058594,		-0.111328,		0.457031),	\
		half3(-0.115234,		-0.287109,		-0.259766)	\
	};
	
	
float my_getLinearDepth(sampler2D tex, float2 tc)
{
	//return getLinearDepthFromRenderedZ(tex, tc);
	
	//from linear Z:
	return tex2D(tex, tc) * fFarClipPlane;	////TODO: need to verify how our linear depth is different from Crytek for the algorithm to work correctly
	//this (wrong) depth value influences on the radius (which way too big) and the distScale (eliminating important samples)
}


float4 ssbn_accumulate(float4 baseTC)
{
	half4 normalDiffOut = (half4)0;
	
	float3x3 matViewProjection_nrm = (float3x3)matViewProjection;
	float3x3 matViewProjectionInv_nrm = (float3x3)g_mat_view_inv; 
	
	//get linear depth
	const float fCenterDepth = my_getLinearDepth(smp_depth, baseTC.xy);
	
	half3 vNormalWS = normalize(tex2D(smp_normal, baseTC.xy) * 2.0h - 1.0h);
		
	if(fCenterDepth < 2000.0)
	{
		//get normal in view space
		half3 vNormalVS = mul(vNormalWS, matViewProjection_nrm);
		
		float2 noise_texture_size = float2(4,4);
		
		const half3 vJitteringVector = tex2D(smp_noise, baseTC.xy * g_resolution.xy / noise_texture_size).xyz * 2.0h - 1.0h;	//get jittering vector for dithering
		const half3 radiusParams = half3(0.4, 0.02, 0.6);
		const half radius = clamp( radiusParams.x / fCenterDepth, radiusParams.y, radiusParams.z );
		const half3 kernel[32] = KERNEL_SAMPLES_32;
		const half effectAmount  = 3.h;
		
		//sampling loop		
		half4 sumVisibility = (half4)0;		
		half3 avgVisibleNormDir = (half3)0;
		half3 avgAllNormDir = (half3)0;		//kept for debug, should be ~= vNormalVS
		
		//HINT_UNROLL
		for (int i = 0; i < OCCLUSION_SAMPLE_COUNT; i += 4)
		{
			//*TODO* use our functions instead of kernel[], reflect and hemisphere-dot
			half3 vSample[4];
			vSample[0] = kernel[i+0];
			vSample[1] = kernel[i+1];
			vSample[2] = kernel[i+2];
			vSample[3] = kernel[i+3];
		
			// Reflect the sample around the dithering normal
			vSample[0] = reflect(vSample[0], vJitteringVector);
			vSample[1] = reflect(vSample[1], vJitteringVector);
			vSample[2] = reflect(vSample[2], vJitteringVector);
			vSample[3] = reflect(vSample[3], vJitteringVector);

			// Make sure that the sample is in the hemisphere defined by the pixel normal
			vSample[0] = ((dot(vNormalVS, vSample[0]) >= 0.0f) ? vSample[0] : -vSample[0]);
			vSample[1] = ((dot(vNormalVS, vSample[1]) >= 0.0f) ? vSample[1] : -vSample[1]);
			vSample[2] = ((dot(vNormalVS, vSample[2]) >= 0.0f) ? vSample[2] : -vSample[2]);
			vSample[3] = ((dot(vNormalVS, vSample[3]) >= 0.0f) ? vSample[3] : -vSample[3]);

			half4 fSampleDepth = half4(
				vSample[0].z,
				vSample[1].z,
				vSample[2].z,
				vSample[3].z );
				
			half4	fTapDepth = half4(
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[0].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[1].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[2].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[3].xy)
			);
			
			fTapDepth = fTapDepth / fCenterDepth;
			
			// Compute the relative sample depth. The depth is multiplied by 2 in order to avoid the sampling sphere
			// distortion since the screen space is in [0..1]x[0..1] while the depth is in [-1..1].
			half4 distScale = (1.h + fSampleDepth * 2.h - fTapDepth) / radius;
		
			// Normalize sample vectors into rays
			vSample[0] = normalize( vSample[0] );
			vSample[1] = normalize( vSample[1] );
			vSample[2] = normalize( vSample[2] );
			vSample[3] = normalize( vSample[3] );
		
		
			half4 fadeOut = saturate( 1.h / distScale );
			half4 dirFadeOut = half4(
				dot( vSample[0], vNormalVS ),
				dot( vSample[1], vNormalVS ),
				dot( vSample[2], vNormalVS ),
				dot( vSample[3], vNormalVS )
			);			
			dirFadeOut = saturate( dirFadeOut );
		
			// Cumulate visbility and rays
			sumVisibility += fadeOut * dirFadeOut;

			half4 dirStrength = (1 - fadeOut) * dirFadeOut;
			avgVisibleNormDir  += vSample[0] * dirStrength.x
								+ vSample[1] * dirStrength.y
								+ vSample[2] * dirStrength.z
								+ vSample[3] * dirStrength.w;

			avgAllNormDir	+= vSample[0]
							 + vSample[1]
							 + vSample[2]
							 + vSample[3];
		}
		
		//final cumulation
		sumVisibility *= effectAmount / OCCLUSION_SAMPLE_COUNT;
		half finalVisibility = dot( 1, sumVisibility );
		
		avgVisibleNormDir = normalize(avgVisibleNormDir);
		avgAllNormDir = normalize(avgAllNormDir);
		
		half3 occlusionAmount = half3(finalVisibility, finalVisibility, finalVisibility);
		
		half3 scaleOfNormals = finalVisibility * vNormalVS;
		scaleOfNormals = mul(scaleOfNormals, matViewProjectionInv_nrm);
		
		half3 diffOfNormals = avgAllNormDir - avgVisibleNormDir;
		diffOfNormals = mul(-diffOfNormals, matViewProjectionInv_nrm);
		
		avgVisibleNormDir = mul(avgVisibleNormDir, matViewProjectionInv_nrm);
		avgAllNormDir = mul(avgAllNormDir, matViewProjectionInv_nrm);
		
		return float4(occlusionAmount, 1);
		return float4(scaleOfNormals * 0.5 + 0.5, 1);
	}	
	return float4(0,0,0,1);

}


float4 ssbn_blur(float2 tex, float2 dir) : COLOR
{
	float weights[9] =
	{
		0.013519569015984728,
		0.047662179108871855,
		0.11723004402070096,
		0.20116755999375591,
		0.240841295721373,
		0.20116755999375591,
		0.11723004402070096,
		0.047662179108871855,
		0.013519569015984728
	};

	float indices[9] = {-4, -3, -2, -1, 0, +1, +2, +3, +4};

	float2 step = dir/g_resolution.xy;

	float3 normal[9];

	normal[0] = tex2D(smp_normal, tex + indices[0]*step).xyz;
	normal[1] = tex2D(smp_normal, tex + indices[1]*step).xyz;
	normal[2] = tex2D(smp_normal, tex + indices[2]*step).xyz;
	normal[3] = tex2D(smp_normal, tex + indices[3]*step).xyz;
	normal[4] = tex2D(smp_normal, tex + indices[4]*step).xyz;
	normal[5] = tex2D(smp_normal, tex + indices[5]*step).xyz;
	normal[6] = tex2D(smp_normal, tex + indices[6]*step).xyz;
	normal[7] = tex2D(smp_normal, tex + indices[7]*step).xyz;
	normal[8] = tex2D(smp_normal, tex + indices[8]*step).xyz;

	float total_weight = 1.0;
	float discard_threshold = 0.85;

	int i;

	for( i=0; i<9; ++i )
	{
		if( dot(normal[i], normal[4]) < discard_threshold )
		{
			total_weight -= weights[i];
			weights[i] = 0;
		}
	}

	//

	float4 res = 0;

	for( i=0; i<9; ++i )
	{
		res += tex2D(smp_occlusion, tex + indices[i]*step) * weights[i];
	}

	res /= total_weight;

	return res;
}
