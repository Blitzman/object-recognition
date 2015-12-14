#ifndef POINTCLOUDOPERATIONSCUDA_H_
#define POINTCLOUDOPERATIONSCUDA_H_

#include <iostream>
#include <limits>
#include <pcl/gpu/kinfu/pixel_rgb.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/containers/device_memory.h>

#include "cuda_runtime_api.h"

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  pcl::gpu::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  pcl::gpu::___cudaSafeCall(expr, __FILE__, __LINE__)    
#endif

namespace pcl
{
	namespace device
	{
		typedef unsigned short ushort;
		typedef DeviceArray2D<float> MapArr;
		typedef DeviceArray2D<ushort> DepthMap;
		typedef float4 PointType;

		void createVMap (const float fx_, const float fy_, const DepthMap & depth, MapArr & vmap);
		float computeCloudResolution(const MapArr& vmap, int rows, int cols);
		void computeNormalsEigen (const MapArr& vmap, MapArr& nmap);
		void bilateralFilter (const DepthMap& src, DepthMap& dst);
	}
}

namespace pcl
{
	namespace gpu
	{
		static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
		{
		    if (cudaSuccess != err)
			error(cudaGetErrorString(err), file, line, func);
		}        

		static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }
	}

	namespace device
	{
		using pcl::gpu::divUp;        

		inline void sync()
		{
			cudaSafeCall(cudaDeviceSynchronize());
		}
	}
}
#endif
