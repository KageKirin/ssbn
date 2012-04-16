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
	{	float3(-0.556641,		-0.037109,		-0.654297),	\
		float3(0.173828,			0.111328,		0.064453),	\
		float3(0.001953,			0.082031,		-0.060547),	\
		float3(0.220703,			-0.359375,		-0.062500),	\
		float3(0.242188,			0.126953,		-0.250000),	\
		float3(0.070313,			-0.025391,		0.148438),	\
		float3(-0.078125,		0.013672,		-0.314453),	\
		float3(0.117188,			-0.140625,		-0.199219),	\
		float3(-0.251953,		-0.558594,		0.082031),	\
		float3(0.308594,			0.193359,		0.324219),	\
		float3(0.173828,			-0.140625,		0.031250),	\
		float3(0.179688,			-0.044922,		0.046875),	\
		float3(-0.146484,		-0.201172,		-0.029297),	\
		float3(-0.300781,		0.234375,		0.539063),	\
		float3(0.228516,			0.154297,		-0.119141),	\
		float3(-0.119141,		-0.003906,		-0.066406),	\
		float3(-0.218750,		0.214844,		-0.250000),	\
		float3(0.113281,			-0.091797,		0.212891),	\
		float3(0.105469,			-0.039063,		-0.019531),	\
		float3(-0.705078,		-0.060547,		0.023438),	\
		float3(0.021484,			0.326172,		0.115234),	\
		float3(0.353516,			0.208984,		-0.294922),	\
		float3(-0.029297,		-0.259766,		0.089844),	\
		float3(-0.240234,		0.146484,		-0.068359),	\
		float3(-0.296875,		0.410156,		-0.291016),	\
		float3(0.078125,			0.113281,		-0.126953),	\
		float3(-0.152344,		-0.019531,		0.142578),	\
		float3(-0.214844,		-0.175781,		0.191406),	\
		float3(0.134766,			0.414063,		-0.707031),	\
		float3(0.291016,			-0.833984,		-0.183594),	\
		float3(-0.058594,		-0.111328,		0.457031),	\
		float3(-0.115234,		-0.287109,		-0.259766)	\
	};

	
#define DSSDO_KERNEL_SAMPLES_32		\
	{	float3(-0.134,			0.044,			-0.825),	\
		float3(0.045,			-0.431,			-0.529),	\
		float3(-0.537,			0.195,			-0.371),	\
		float3(0.525,			-0.397,			0.713),		\
		float3(0.895,			0.302,			0.139),		\
		float3(-0.613,			-0.408,			-0.141),	\
		float3(0.307,			0.822,			0.169),		\
		float3(-0.819,			0.037,			-0.388),	\
		float3(0.376,			0.009,			0.193),		\
		float3(-0.006,			-0.103,			-0.035),	\
		float3(0.098,			0.393,			0.019),		\
		float3(0.542,			-0.218,			-0.593),	\
		float3(0.526,			-0.183,			0.424),		\
		float3(-0.529,			-0.178,			0.684),		\
		float3(0.066,			-0.657,			-0.570),	\
		float3(-0.214,			0.288,			0.188),		\
		float3(-0.689,			-0.222,			-0.192),	\
		float3(-0.008,			-0.212,			-0.721),	\
		float3(0.053,			-0.863,			0.054),		\
		float3(0.639,			-0.558,			0.289),		\
		float3(-0.255,			0.958,			0.099),		\
		float3(-0.488,			0.473,			-0.381),	\
		float3(-0.592,			-0.332,			0.137),		\
		float3(0.080,			0.756,			-0.494),	\
		float3(-0.638,			0.319,			0.686),		\
		float3(-0.663,			0.230,			-0.634),	\
		float3(0.235,			-0.547,			0.664),		\
		float3(0.164,			-0.710,			0.086),		\
		float3(-0.009,			0.493,			-0.038),	\
		float3(-0.322,			0.147,			-0.105),	\
		float3(-0.554,			-0.725,			0.289),		\
		float3(0.534,			0.157,			-0.250)		\
	};
	
float my_getLinearDepth(sampler2D tex, float2 tc)
{
	//float3 eye_pos = g_mat_view_inv[3].xyz;
	//float3 sample_pos = tex2D(tex, tc);	
	//return distance(eye_pos, sample_pos)/ fFarClipPlane;
	
	//return getLinearDepthFromRenderedZ(tex, tc);
	
	//from linear Z:
	return tex2D(tex, tc) * fFarClipPlane;	////TODO: need to verify how our linear depth is different from Crytek for the algorithm to work correctly
	//this (wrong) depth value influences on the radius (which way too big) and the distScale (eliminating important samples)
}


float4 ssbn_accumulate(float4 baseTC)
{
	float4 normalDiffOut = (float4)0;
	
	float3x3 matViewProjection_nrm = (float3x3)matViewProjection;
	float3x3 matViewProjectionInv_nrm = (float3x3)g_mat_view_inv; 
	
	//get linear depth
	const float fCenterDepth = my_getLinearDepth(smp_depth, baseTC.xy);	//smp_depth		smp_position
	
	float3 vNormalWS = normalize(tex2D(smp_normal, baseTC.xy) * 2.0h - 1.0h);
		
	if(fCenterDepth < 2000.0)
	{
		//get normal in view space
		float3 vNormalVS = mul(vNormalWS, matViewProjection_nrm);
		
		float2 noise_texture_size = float2(4,4);
		
		const float3 vJitteringVector = tex2D(smp_noise, baseTC.xy * 2 *g_resolution.xy / noise_texture_size).xyz * 2.0h - 1.0h;	//get jittering vector for dithering
		const float3 radiusParams = float3(0.002, 0.001, 0.02);		// / fCenterDepth
		const float radius = clamp( radiusParams.x, radiusParams.y, radiusParams.z );
		const float3 kernel[32] = KERNEL_SAMPLES_32;
		const float effectAmount  = 3.h;
		
		//sampling loop		
		float4 sumVisibility = (float4)0;		
		float3 avgVisibleNormDir = (float3)0;
		float3 avgAllNormDir = (float3)0;		//kept for debug, should be ~= vNormalVS
		
		//HINT_UNROLL
		for (int i = 0; i < OCCLUSION_SAMPLE_COUNT; i += 4)
		{
			//*TODO* use our functions instead of kernel[], reflect and hemisphere-dot
			float3 vSample[4];
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

			float4 fSampleDepth = float4(
				vSample[0].z,
				vSample[1].z,
				vSample[2].z,
				vSample[3].z );
				
			float4	fTapDepth = float4(
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[0].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[1].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[2].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + vSample[3].xy)
			);
			
			fTapDepth = fTapDepth / fCenterDepth;
			
			// Compute the relative sample depth. The depth is multiplied by 2 in order to avoid the sampling sphere
			// distortion since the screen space is in [0..1]x[0..1] while the depth is in [-1..1].
			float4 distScale = (1.h + fSampleDepth * 2.h - fTapDepth) / radius;
		
			// Normalize sample vectors into rays
			vSample[0] = normalize( vSample[0] );
			vSample[1] = normalize( vSample[1] );
			vSample[2] = normalize( vSample[2] );
			vSample[3] = normalize( vSample[3] );
		
		
			float4 fadeOut = saturate( 1.h / distScale );
			float4 dirFadeOut = float4(
				dot( vSample[0], vNormalVS ),
				dot( vSample[1], vNormalVS ),
				dot( vSample[2], vNormalVS ),
				dot( vSample[3], vNormalVS )
			);			
			dirFadeOut = saturate( dirFadeOut );
		
			// Cumulate visbility and rays
			sumVisibility += fadeOut * dirFadeOut;

			float4 dirStrength = (1 - fadeOut) * dirFadeOut;
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
		float finalVisibility = dot( 1, sumVisibility );
		
		avgVisibleNormDir = normalize(avgVisibleNormDir);
		avgAllNormDir = normalize(avgAllNormDir);
		
		float3 occlusionAmount = float3(finalVisibility, finalVisibility, finalVisibility);
		
		float3 scaleOfNormals = finalVisibility * vNormalVS;
		scaleOfNormals = mul(scaleOfNormals, matViewProjectionInv_nrm);
		
		float3 diffOfNormals = vNormalVS - avgVisibleNormDir;
		diffOfNormals = mul(diffOfNormals, matViewProjectionInv_nrm);
		
		avgVisibleNormDir = mul(avgVisibleNormDir, matViewProjectionInv_nrm);
		avgAllNormDir = mul(avgAllNormDir, matViewProjectionInv_nrm);
		
		//return float4(occlusionAmount, 1);
		//return float4(-scaleOfNormals * 0.5 + 0.5, 1);
		return float4(diffOfNormals * 0.5 + 0.5, 1);
		//return float4(avgVisibleNormDir * 0.5 + 0.5, 1);
		//return float4(avgAllNormDir * 0.5 + 0.5, 1);
		//return float4(mul(vNormalVS, matViewProjectionInv_nrm) * 0.5 + 0.5, 1);
		
		
		//return float4(-scaleOfNormals, 1);
		//return float4(diffOfNormals, 1);
		//return float4(avgVisibleNormDir, 1);
		//return float4(avgAllNormDir, 1);
		//return float4(mul(vNormalVS, matViewProjectionInv_nrm), 1);
	}	
	return float4(0.5, 0.5, 0.5, 1);
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
