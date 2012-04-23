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
		float3(0.173828,		0.111328,		0.064453),	\
		float3(0.001953,		0.082031,		-0.060547),	\
		float3(0.220703,		-0.359375,		-0.062500),	\
		float3(0.242188,		0.126953,		-0.250000),	\
		float3(0.070313,		-0.025391,		0.148438),	\
		float3(-0.078125,		0.013672,		-0.314453),	\
		float3(0.117188,		-0.140625,		-0.199219),	\
		float3(-0.251953,		-0.558594,		0.082031),	\
		float3(0.308594,		0.193359,		0.324219),	\
		float3(0.173828,		-0.140625,		0.031250),	\
		float3(0.179688,		-0.044922,		0.046875),	\
		float3(-0.146484,		-0.201172,		-0.029297),	\
		float3(-0.300781,		0.234375,		0.539063),	\
		float3(0.228516,		0.154297,		-0.119141),	\
		float3(-0.119141,		-0.003906,		-0.066406),	\
		float3(-0.218750,		0.214844,		-0.250000),	\
		float3(0.113281,		-0.091797,		0.212891),	\
		float3(0.105469,		-0.039063,		-0.019531),	\
		float3(-0.705078,		-0.060547,		0.023438),	\
		float3(0.021484,		0.326172,		0.115234),	\
		float3(0.353516,		0.208984,		-0.294922),	\
		float3(-0.029297,		-0.259766,		0.089844),	\
		float3(-0.240234,		0.146484,		-0.068359),	\
		float3(-0.296875,		0.410156,		-0.291016),	\
		float3(0.078125,		0.113281,		-0.126953),	\
		float3(-0.152344,		-0.019531,		0.142578),	\
		float3(-0.214844,		-0.175781,		0.191406),	\
		float3(0.134766,		0.414063,		-0.707031),	\
		float3(0.291016,		-0.833984,		-0.183594),	\
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
	const float3 eye_pos = g_mat_view_inv[3].xyz;
	float3 sample_pos = tex2D(tex, tc);	
	return distance(eye_pos, sample_pos);	/// fFarClipPlane;
	
	//return getLinearDepthFromRenderedZ(tex, tc);
	
	//from linear Z:
	//return tex2D(tex, tc) * fFarClipPlane;	//this (wrong) depth value influences on the radius (which way too big) and the distScale (eliminating important samples)
}


float4 ssbn_accumulate(float4 baseTC)
{
	float3x3 matViewProjection_nrm = (float3x3)matViewProjection;
	float3x3 matViewProjectionInv_nrm = (float3x3)g_mat_view_inv; 
	
	//get linear depth
	const float Zcenter = my_getLinearDepth(smp_depth, baseTC.xy);	//smp_depth		smp_position
	const float Zcenter_inv = 1.0 / Zcenter;
	const half ZthresholdOut	= 1.5;	//threshold, to be parametrized

	float3 Ncenter_WS = normalize(tex2D(smp_normal, baseTC.xy) * 2.0h - 1.0h);
		
	if(Zcenter < 200.0)
	{
		//get normal in view space
		float3 Ncenter_VS = mul(Ncenter_WS, matViewProjection_nrm);
		
		static const float2 noise_texture_size = float2(4,4);
		
		const float3 Njitter = tex2D(smp_noise, baseTC.xy * g_resolution.xy / noise_texture_size).xyz * 2.0h - 1.0h;	//get jittering vector for dithering
		const float3 radiusParams = float3(0.4, 0.02, 0.06);		// 
		const float radius = clamp( radiusParams.x * Zcenter_inv, radiusParams.y, radiusParams.z );
		
		const float3 kernel[32] = KERNEL_SAMPLES_32;
		
		//sampling loop		
		half4 sumVisibility = (half4)0;
		half4 sumValidity = (half4)0;
		half3 avgVisibleNormDir = (half3)0;
		half3 avgAllNormDir = (half3)0;

		for (int i = 0; i < OCCLUSION_SAMPLE_COUNT; i += 4)
		{
			//vectorized, processing 4 samples at once for speed
			half3 sampleDir[4];
			sampleDir[0] = kernel[i+0] * radius;
			sampleDir[1] = kernel[i+1] * radius;
			sampleDir[2] = kernel[i+2] * radius;
			sampleDir[3] = kernel[i+3] * radius;		

			// Reflect the sample around the dithering normal
			sampleDir[0] = reflect(sampleDir[0], Njitter);
			sampleDir[1] = reflect(sampleDir[1], Njitter);
			sampleDir[2] = reflect(sampleDir[2], Njitter);
			sampleDir[3] = reflect(sampleDir[3], Njitter);

			// Make sure that the sample is in the hemisphere defined by the pixel normal
			sampleDir[0] *= sign(dot(Ncenter_VS, sampleDir[0]));
			sampleDir[1] *= sign(dot(Ncenter_VS, sampleDir[1]));
			sampleDir[2] *= sign(dot(Ncenter_VS, sampleDir[2]));
			sampleDir[3] *= sign(dot(Ncenter_VS, sampleDir[3]));

			half4 ZsampleDir = half4(
				sampleDir[0].z,
				sampleDir[1].z,
				sampleDir[2].z,
				sampleDir[3].z );			

			half4	ZsampleTap = half4(
				my_getLinearDepth(smp_depth, baseTC.xy + sampleDir[0].xy),	//shouldn't we transform project the sampleDirs into texture space? (unless they are already)
				my_getLinearDepth(smp_depth, baseTC.xy + sampleDir[1].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + sampleDir[2].xy),
				my_getLinearDepth(smp_depth, baseTC.xy + sampleDir[3].xy)
			);
			ZsampleTap -= Zcenter;			

			half3 sampleTap[4];
			sampleTap[0] = half3(sampleDir[0].xy, ZsampleTap.x);
			sampleTap[1] = half3(sampleDir[1].xy, ZsampleTap.y);
			sampleTap[2] = half3(sampleDir[2].xy, ZsampleTap.z);
			sampleTap[3] = half3(sampleDir[3].xy, ZsampleTap.w);			

			half4 DsampleTap = half4(
				length(sampleTap[0]),
				length(sampleTap[1]),
				length(sampleTap[2]),
				length(sampleTap[3])
			);
			
			sampleTap[0] /= DsampleTap.x;
			sampleTap[1] /= DsampleTap.y;
			sampleTap[2] /= DsampleTap.z;
			sampleTap[3] /= DsampleTap.w;			

			half4 visibility = half4(
				dot(sampleTap[0], Ncenter_VS),
				dot(sampleTap[1], Ncenter_VS),
				dot(sampleTap[2], Ncenter_VS),
				dot(sampleTap[3], Ncenter_VS)
			);
			visibility = saturate(visibility);

			half4 validity = saturate( -sign( abs(DsampleTap) - ZthresholdOut ) );	//valid if(abs(ZsampleTap) < ZthresholdOut)
			half4 sampleWeight = validity * visibility;
			
			// Normalize sample vectors into rays
			sampleDir[0] = normalize( sampleDir[0] );
			sampleDir[1] = normalize( sampleDir[1] );
			sampleDir[2] = normalize( sampleDir[2] );
			sampleDir[3] = normalize( sampleDir[3] );
		
			// Cumulate visbility and rays
			sumVisibility += sampleWeight;
			avgVisibleNormDir  += sampleDir[0] * sampleWeight.x
								+ sampleDir[1] * sampleWeight.y
								+ sampleDir[2] * sampleWeight.z
								+ sampleDir[3] * sampleWeight.w;
								
			avgAllNormDir 	+=	sampleDir[0]
							+	sampleDir[1]
							+	sampleDir[2]
							+	sampleDir[3];
		}
		
		//final cumulation
		half finalVisibility = dot( 1, sumVisibility );
		finalVisibility /= OCCLUSION_SAMPLE_COUNT;
		finalVisibility = finalVisibility;
		
		half3 finalVisibleNormDir = avgAllNormDir - avgVisibleNormDir;
		finalVisibleNormDir /= dot( 1, sumVisibility );
		
		
		
		float3 occlusionAmount = float3(finalVisibility, finalVisibility, finalVisibility);
		
		//float3 scaleOfNormals = finalVisibility * Ncenter_VS;
		//scaleOfNormals = mul(scaleOfNormals, matViewProjectionInv_nrm);
		
		float3 diffOfNormals = Ncenter_VS - avgVisibleNormDir;
		diffOfNormals = mul(diffOfNormals, matViewProjectionInv_nrm);
		
		avgVisibleNormDir = mul(avgVisibleNormDir, matViewProjectionInv_nrm);
		//avgAllNormDir = mul(avgAllNormDir, matViewProjectionInv_nrm);
		
		//return float4(occlusionAmount, 1);
		//return float4(-scaleOfNormals * 0.5 + 0.5, 1);
		return float4(diffOfNormals * 0.5 + 0.5, finalVisibility);
		//return float4(avgVisibleNormDir * 0.5 + 0.5, 1);
		//return float4(avgAllNormDir * 0.5 + 0.5, 1);
		//return float4(mul(Ncenter_VS, matViewProjectionInv_nrm) * 0.5 + 0.5, 1);
	}	
	return float4(0.5, 0.5, 0.5, 0);	//return a 0 vector
}
