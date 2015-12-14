#include "PointCloudOperationsCUDA.h"

using namespace pcl::device;
using namespace pcl::gpu;

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 8

namespace pcl
{
	namespace device
	{
		template<typename T> struct numeric_limits;

		template<> struct numeric_limits<float>
		{
			__device__ __forceinline__ static float 
			quiet_NaN() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
			__device__ __forceinline__ static float 
			epsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };

			__device__ __forceinline__ static float 
			min() { return 1.175494351e-38f/*FLT_MIN*/; };
			__device__ __forceinline__ static float 
			max() { return 3.402823466e+38f/*FLT_MAX*/; };
		};

		template<> struct numeric_limits<short>
		{
			__device__ __forceinline__ static short 
			max() { return SHRT_MAX; };
		};

		__device__ __forceinline__ float
		dot(const float3& v1, const float3& v2)
		{
			return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
		}

		__device__ __forceinline__ float3&
		operator+=(float3& vec, const float& v)
		{
			vec.x += v;  vec.y += v;  vec.z += v; return vec;
		}

		__device__ __forceinline__ float3
		operator+(const float3& v1, const float3& v2)
		{
			return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
		}

		__device__ __forceinline__ float3&
		operator*=(float3& vec, const float& v)
		{
			vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
		}

		__device__ __forceinline__ float3
		operator-(const float3& v1, const float3& v2)
		{
			return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
		}

		__device__ __forceinline__ float3
		operator*(const float3& v1, const float& v)
		{
			return make_float3(v1.x * v, v1.y * v, v1.z * v);
		}

		__device__ __forceinline__ float
		norm(const float3& v)
		{
			return sqrt(dot(v, v));
		}

		__device__ __forceinline__ float3
		normalized(const float3& v)
		{
			return v * rsqrt(dot(v, v));
		}

		__device__ __host__ __forceinline__ float3 
		cross(const float3& v1, const float3& v2)
		{
			return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
		}

		__device__ __forceinline__ float
		distance(const float3& v1, const float3& v2)
		{
			return sqrtf ( (v2.x-v1.x)*(v2.x-v1.x) + (v2.y-v1.y)*(v2.y-v1.y) + (v2.z-v1.z)*(v2.z-v1.z) ) ;
		}

template <class T> 
    __device__ __host__ __forceinline__ void swap ( T& a, T& b )
    {
      T c(a); a=b; b=c;
    }

   __device__ __forceinline__ void computeRoots2(const float& b, const float& c, float3& roots)
     {
       roots.x = 0.f;
       float d = b * b - 4.f * c;
       if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
         d = 0.f;

       float sd = sqrtf(d);

       roots.z = 0.5f * (b + sd);
       roots.y = 0.5f * (b - sd);
     }

__device__ __forceinline__ void 
     computeRoots3(float c0, float c1, float c2, float3& roots)
     {
       if ( fabsf(c0) < numeric_limits<float>::epsilon())// one root is 0 -> quadratic equation
       {
         computeRoots2 (c2, c1, roots);
       }
       else
       {
         const float s_inv3 = 1.f/3.f;
         const float s_sqrt3 = sqrtf(3.f);
         // Construct the parameters used in classifying the roots of the equation
         // and in solving the equation for the roots in closed form.
         float c2_over_3 = c2 * s_inv3;
         float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
         if (a_over_3 > 0.f)
           a_over_3 = 0.f;

         float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

         float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
         if (q > 0.f)
           q = 0.f;

         // Compute the eigenvalues by solving for the roots of the polynomial.
         float rho = sqrtf(-a_over_3);
         float theta = atan2f (sqrtf (-q), half_b)*s_inv3;
         float cos_theta = __cosf (theta);
         float sin_theta = __sinf (theta);
         roots.x = c2_over_3 + 2.f * rho * cos_theta;
         roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
         roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

         // Sort in increasing order.
         if (roots.x >= roots.y)
           swap(roots.x, roots.y);

         if (roots.y >= roots.z)
         {
           swap(roots.y, roots.z);

           if (roots.x >= roots.y)
             swap (roots.x, roots.y);
         }
         if (roots.x <= 0) // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
           computeRoots2 (c2, c1, roots);
       }
     }

struct Eigen33
     {
     public:
       template<int Rows>
       struct MiniMat
       {
         float3 data[Rows];                
         __device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
         __device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
       };
       typedef MiniMat<3> Mat33;
       typedef MiniMat<4> Mat43;
       
       
       static __forceinline__ __device__ float3 
       unitOrthogonal (const float3& src)
       {
         float3 perp;
         /* Let us compute the crossed product of *this with a vector
         * that is not too close to being colinear to *this.
         */

         /* unless the x and y coords are both close to zero, we can
         * simply take ( -y, x, 0 ) and normalize it.
         */
         if(!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
         {   
           float invnm = rsqrtf(src.x*src.x + src.y*src.y);
           perp.x = -src.y * invnm;
           perp.y =  src.x * invnm;
           perp.z = 0.0f;
         }   
         /* if both x and y are close to zero, then the vector is close
         * to the z-axis, so it's far from colinear to the x-axis for instance.
         * So we take the crossed product with (1,0,0) and normalize it. 
         */
         else
         {   
           float invnm = rsqrtf(src.z * src.z + src.y * src.y);
           perp.x = 0.0f;
           perp.y = -src.z * invnm;
           perp.z =  src.y * invnm;
         }   

         return perp;
       }

       __device__ __forceinline__ 
       Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}                      
       __device__ __forceinline__ void 
       compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
       {
         // Scale the matrix so its entries are in [-1,1].  The scaling is applied
         // only when at least one matrix entry has magnitude larger than 1.

         float max01 = fmaxf( fabsf(mat_pkg[0]), fabsf(mat_pkg[1]) );
         float max23 = fmaxf( fabsf(mat_pkg[2]), fabsf(mat_pkg[3]) );
         float max45 = fmaxf( fabsf(mat_pkg[4]), fabsf(mat_pkg[5]) );
         float m0123 = fmaxf( max01, max23);
         float scale = fmaxf( max45, m0123);

         if (scale <= numeric_limits<float>::min())
           scale = 1.f;

         mat_pkg[0] /= scale;
         mat_pkg[1] /= scale;
         mat_pkg[2] /= scale;
         mat_pkg[3] /= scale;
         mat_pkg[4] /= scale;
         mat_pkg[5] /= scale;

         // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
         // eigenvalues are the roots to this equation, all guaranteed to be
         // real-valued, because the matrix is symmetric.
         float c0 = m00() * m11() * m22() 
             + 2.f * m01() * m02() * m12()
             - m00() * m12() * m12() 
             - m11() * m02() * m02() 
             - m22() * m01() * m01();
         float c1 = m00() * m11() - 
             m01() * m01() + 
             m00() * m22() - 
             m02() * m02() + 
             m11() * m22() - 
             m12() * m12();
         float c2 = m00() + m11() + m22();

         computeRoots3(c0, c1, c2, evals);

         if(evals.z - evals.x <= numeric_limits<float>::epsilon())
         {                                   
           evecs[0] = make_float3(1.f, 0.f, 0.f);
           evecs[1] = make_float3(0.f, 1.f, 0.f);
           evecs[2] = make_float3(0.f, 0.f, 1.f);
         }
         else if (evals.y - evals.x <= numeric_limits<float>::epsilon() )
         {
           // first and second equal                
           tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
           tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

           vec_tmp[0] = cross(tmp[0], tmp[1]);
           vec_tmp[1] = cross(tmp[0], tmp[2]);
           vec_tmp[2] = cross(tmp[1], tmp[2]);

           float len1 = dot (vec_tmp[0], vec_tmp[0]);
           float len2 = dot (vec_tmp[1], vec_tmp[1]);
           float len3 = dot (vec_tmp[2], vec_tmp[2]);

           if (len1 >= len2 && len1 >= len3)
           {
             evecs[2] = vec_tmp[0] * rsqrtf (len1);
           }
           else if (len2 >= len1 && len2 >= len3)
           {
             evecs[2] = vec_tmp[1] * rsqrtf (len2);
           }
           else
           {
             evecs[2] = vec_tmp[2] * rsqrtf (len3);
           }

           evecs[1] = unitOrthogonal(evecs[2]);
           evecs[0] = cross(evecs[1], evecs[2]);
         }
         else if (evals.z - evals.y <= numeric_limits<float>::epsilon() )
         {
           // second and third equal                                    
           tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
           tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

           vec_tmp[0] = cross(tmp[0], tmp[1]);
           vec_tmp[1] = cross(tmp[0], tmp[2]);
           vec_tmp[2] = cross(tmp[1], tmp[2]);

           float len1 = dot(vec_tmp[0], vec_tmp[0]);
           float len2 = dot(vec_tmp[1], vec_tmp[1]);
           float len3 = dot(vec_tmp[2], vec_tmp[2]);

           if (len1 >= len2 && len1 >= len3)
           {
             evecs[0] = vec_tmp[0] * rsqrtf(len1);
           }
           else if (len2 >= len1 && len2 >= len3)
           {
             evecs[0] = vec_tmp[1] * rsqrtf(len2);
           }
           else
           {
             evecs[0] = vec_tmp[2] * rsqrtf(len3);
           }

           evecs[1] = unitOrthogonal( evecs[0] );
           evecs[2] = cross(evecs[0], evecs[1]);
         }
         else
         {

           tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
           tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

           vec_tmp[0] = cross(tmp[0], tmp[1]);
           vec_tmp[1] = cross(tmp[0], tmp[2]);
           vec_tmp[2] = cross(tmp[1], tmp[2]);

           float len1 = dot(vec_tmp[0], vec_tmp[0]);
           float len2 = dot(vec_tmp[1], vec_tmp[1]);
           float len3 = dot(vec_tmp[2], vec_tmp[2]);

           float mmax[3];

           unsigned int min_el = 2;
           unsigned int max_el = 2;
           if (len1 >= len2 && len1 >= len3)
           {
             mmax[2] = len1;
             evecs[2] = vec_tmp[0] * rsqrtf (len1);
           }
           else if (len2 >= len1 && len2 >= len3)
           {
             mmax[2] = len2;
             evecs[2] = vec_tmp[1] * rsqrtf (len2);
           }
           else
           {
             mmax[2] = len3;
             evecs[2] = vec_tmp[2] * rsqrtf (len3);
           }

           tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
           tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

           vec_tmp[0] = cross(tmp[0], tmp[1]);
           vec_tmp[1] = cross(tmp[0], tmp[2]);
           vec_tmp[2] = cross(tmp[1], tmp[2]);                    

           len1 = dot(vec_tmp[0], vec_tmp[0]);
           len2 = dot(vec_tmp[1], vec_tmp[1]);
           len3 = dot(vec_tmp[2], vec_tmp[2]);

           if (len1 >= len2 && len1 >= len3)
           {
             mmax[1] = len1;
             evecs[1] = vec_tmp[0] * rsqrtf (len1);
             min_el = len1 <= mmax[min_el] ? 1 : min_el;
             max_el = len1  > mmax[max_el] ? 1 : max_el;
           }
           else if (len2 >= len1 && len2 >= len3)
           {
             mmax[1] = len2;
             evecs[1] = vec_tmp[1] * rsqrtf (len2);
             min_el = len2 <= mmax[min_el] ? 1 : min_el;
             max_el = len2  > mmax[max_el] ? 1 : max_el;
           }
           else
           {
             mmax[1] = len3;
             evecs[1] = vec_tmp[2] * rsqrtf (len3);
             min_el = len3 <= mmax[min_el] ? 1 : min_el;
             max_el = len3 >  mmax[max_el] ? 1 : max_el;
           }

           tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
           tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

           vec_tmp[0] = cross(tmp[0], tmp[1]);
           vec_tmp[1] = cross(tmp[0], tmp[2]);
           vec_tmp[2] = cross(tmp[1], tmp[2]);

           len1 = dot (vec_tmp[0], vec_tmp[0]);
           len2 = dot (vec_tmp[1], vec_tmp[1]);
           len3 = dot (vec_tmp[2], vec_tmp[2]);


           if (len1 >= len2 && len1 >= len3)
           {
             mmax[0] = len1;
             evecs[0] = vec_tmp[0] * rsqrtf (len1);
             min_el = len3 <= mmax[min_el] ? 0 : min_el;
             max_el = len3  > mmax[max_el] ? 0 : max_el;
           }
           else if (len2 >= len1 && len2 >= len3)
           {
             mmax[0] = len2;
             evecs[0] = vec_tmp[1] * rsqrtf (len2);
             min_el = len3 <= mmax[min_el] ? 0 : min_el;
             max_el = len3  > mmax[max_el] ? 0 : max_el; 		
           }
           else
           {
             mmax[0] = len3;
             evecs[0] = vec_tmp[2] * rsqrtf (len3);
             min_el = len3 <= mmax[min_el] ? 0 : min_el;
             max_el = len3  > mmax[max_el] ? 0 : max_el;	  
           }

           unsigned mid_el = 3 - min_el - max_el;
           evecs[min_el] = normalized( cross( evecs[(min_el+1) % 3], evecs[(min_el+2) % 3] ) );
           evecs[mid_el] = normalized( cross( evecs[(mid_el+1) % 3], evecs[(mid_el+2) % 3] ) );
         }
         // Rescale back to the original size.
         evals *= scale;
       }
     private:
       volatile float* mat_pkg;

       __device__  __forceinline__ float m00() const { return mat_pkg[0]; }
       __device__  __forceinline__ float m01() const { return mat_pkg[1]; }
       __device__  __forceinline__ float m02() const { return mat_pkg[2]; }
       __device__  __forceinline__ float m10() const { return mat_pkg[1]; }
       __device__  __forceinline__ float m11() const { return mat_pkg[3]; }
       __device__  __forceinline__ float m12() const { return mat_pkg[4]; }
       __device__  __forceinline__ float m20() const { return mat_pkg[2]; }
       __device__  __forceinline__ float m21() const { return mat_pkg[4]; }
       __device__  __forceinline__ float m22() const { return mat_pkg[5]; }

       __device__  __forceinline__ float3 row0() const { return make_float3( m00(), m01(), m02() ); }
       __device__  __forceinline__ float3 row1() const { return make_float3( m10(), m11(), m12() ); }
       __device__  __forceinline__ float3 row2() const { return make_float3( m20(), m21(), m22() ); }

       __device__  __forceinline__ static bool isMuchSmallerThan (float x, float y)
       {
           // copied from <eigen>/include/Eigen/src/Core/NumTraits.h
           const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon(); 
           return x * x <= prec_sqr * y * y;
       }
     };  

		enum
		{
			kx = 7,
			ky = 7,
			STEP = 1
		};

		__global__ void
		computeVmapKernel (const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy)
		{
			int u = threadIdx.x + blockIdx.x * blockDim.x;
			int v = threadIdx.y + blockIdx.y * blockDim.y;

			if (u < depth.cols && v < depth.rows)
			{
				float z = depth.ptr (v)[u] / 1000.f; // load and convert: mm -> meters

				if (z != 0)
				{
					float vx = z * (u - cx) * fx_inv;
					float vy = z * (v - cy) * fy_inv;
					float vz = z;

					vmap.ptr (v                 )[u] = vx;
					vmap.ptr (v + depth.rows    )[u] = vy;
					vmap.ptr (v + depth.rows * 2)[u] = vz;
				}
				else
				{
					vmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
				}
			}
		}

		__global__ void
		computeCloudResolutionKernel (int rows, int cols, const PtrStep<float> vmap, float *nearest_neighbor)
    		{
			int u = threadIdx.x + blockIdx.x * blockDim.x;
      			int v = threadIdx.y + blockIdx.y * blockDim.y;

			if (u >= cols || v >= rows)
        			return;

			int global_index = (v*cols+u);

			float v_i_x = vmap.ptr (v)[u];
			float v_i_y = vmap.ptr (v +  rows)[u];
			float v_i_z = vmap.ptr (v +  2*rows)[u];
			float3 centroid = make_float3( v_i_x, v_i_y, v_i_z);

			int ty = min (v - ky / 2 + ky, rows - 1);
			int tx = min (u - kx / 2 + kx, cols - 1);

			float min_distance=numeric_limits<float>::max();
			for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
			{
				for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
				{
					float v_x = vmap.ptr (cy)[cx];
					if (!isnan (v_x) && cy != v && cx != u)
					{
						float v_y = vmap.ptr (cy + rows)[cx];
						float v_z = vmap.ptr (cy + 2 * rows)[cx];
						float3 v = make_float3(v_x,v_y,v_z) - centroid;
						float aux = norm(v);
						if( aux < min_distance )
							min_distance = aux;
					}
				}
			}

			nearest_neighbor[global_index]=min_distance;
		}

		__global__ void
		computeNmapKernelEigen (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
		{
			int u = threadIdx.x + blockIdx.x * blockDim.x;
			int v = threadIdx.y + blockIdx.y * blockDim.y;

			if (u >= cols || v >= rows)
				return;

			nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

			if (isnan (vmap.ptr (v)[u]))
				return;

			int ty = min (v - ky / 2 + ky, rows - 1);
			int tx = min (u - kx / 2 + kx, cols - 1);

			float3 centroid = make_float3 (0.f, 0.f, 0.f);
			int counter = 0;
			
			for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
			{
				for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
				{
					float v_x = vmap.ptr (cy)[cx];
					if (!isnan (v_x))
					{
						centroid.x += v_x;
						centroid.y += vmap.ptr (cy + rows)[cx];
						centroid.z += vmap.ptr (cy + 2 * rows)[cx];
						++counter;
					}
				}
			}

			if (counter < kx * ky / 2)
				return;

			centroid *= 1.f / counter;

			float cov[] = {0, 0, 0, 0, 0, 0};

			for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
			{
				for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
				{
					float3 v;
					v.x = vmap.ptr (cy)[cx];
					if (isnan (v.x))
						continue;

					v.y = vmap.ptr (cy + rows)[cx];
					v.z = vmap.ptr (cy + 2 * rows)[cx];

					float3 d = v - centroid;

					cov[0] += d.x * d.x;               //cov (0, 0)
					cov[1] += d.x * d.y;               //cov (0, 1)
					cov[2] += d.x * d.z;               //cov (0, 2)
					cov[3] += d.y * d.y;               //cov (1, 1)
					cov[4] += d.y * d.z;               //cov (1, 2)
					cov[5] += d.z * d.z;               //cov (2, 2)
				}
			}

			typedef Eigen33::Mat33 Mat33;
			Eigen33 eigen33 (cov);

			Mat33 tmp;
			Mat33 vec_tmp;
			Mat33 evecs;
			float3 evals;
			eigen33.compute (tmp, vec_tmp, evecs, evals);

			float3 n = normalized (evecs[0]);

			u = threadIdx.x + blockIdx.x * blockDim.x;
			v = threadIdx.y + blockIdx.y * blockDim.y;

			// See if we need to flip any plane normals
			float3 pointview = make_float3 (0.f,0.f,0.f);
			float3 p = make_float3 ( vmap.ptr (v)[u], vmap.ptr (v+rows)[u], vmap.ptr (v+rows*2)[u]);

			// Dot product between the (viewpoint - point) and the plane normal
			pointview = pointview-p;
			float cos_theta = dot(pointview,n);

			// Flip the plane normal
			if (cos_theta < 0)
			{
				n.x *= -1;
				n.y *= -1;
				n.z *= -1;
			}

			if( (u == 100 && v == 300) || (u == 101 && v == 300)) 
			{
				//printf(" x:%f y:%f z:%f \n",n.x,n.y,n.z);
			}

			nmap.ptr (v       )[u] = n.x;
			nmap.ptr (v + rows)[u] = n.y;
			nmap.ptr (v + 2 * rows)[u] = n.z;
		}

		const float sigma_color = 30;     //in mm
		const float sigma_space = 4.5;     // in pixels

		__global__ void
		bilateralKernel (const PtrStepSz<ushort> src, PtrStep<ushort> dst, float sigma_space2_inv_half, float sigma_color2_inv_half)
		{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= src.cols || y >= src.rows)
				return;

			const int R = 6;       //static_cast<int>(sigma_space * 1.5);
			const int D = R * 2 + 1;

			int value = src.ptr (y)[x];

			int tx = min (x - D / 2 + D, src.cols - 1);
			int ty = min (y - D / 2 + D, src.rows - 1);

			float sum1 = 0;
			float sum2 = 0;

			for (int cy = max (y - D / 2, 0); cy < ty; ++cy)
			{
				for (int cx = max (x - D / 2, 0); cx < tx; ++cx)
				{
					int tmp = src.ptr (cy)[cx];

					float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
					float color2 = (value - tmp) * (value - tmp);

					float weight = __expf (-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

					sum1 += tmp * weight;
					sum2 += weight;
				}
			}

			int res = __float2int_rn (sum1 / sum2);
			dst.ptr (y)[x] = max (0, min (res, numeric_limits<short>::max ()));
		}

		void createVMap (const float fx_, const float fy_, const DepthMap & depth, MapArr & vmap)
		{
			//std::cout << "Launching cloud projection kernel...\n";

			vmap.create (depth.rows () * 3, depth.cols ());

			dim3 block (BLOCKDIM_X, BLOCKDIM_Y);
			dim3 grid (1, 1, 1);
			grid.x = divUp (depth.cols (), block.x);
			grid.y = divUp (depth.rows (), block.y);

			//std::cout << "Block size (" << block.x << ", " << block.y << ", " << block.z << ")\n";
			//std::cout << "Grid size (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

			float fx = fx_, cx = depth.cols() >> 1;
			float fy = fy_, cy = depth.rows() >> 1;

			computeVmapKernel<<<grid, block>>>(depth, vmap, fx, fy, cx, cy);
			cudaSafeCall(cudaGetLastError());
		}

		float computeCloudResolution(const MapArr& vmap, int rows, int cols)
		{
			//std::cout << "Launching cloud resolution kernel...\n";

			dim3 block(BLOCKDIM_X, BLOCKDIM_Y);
			dim3 grid(1,1,1);
			grid.x = divUp (cols, block.x);
		  	grid.y = divUp (rows, block.y);

			//std::cout << "Block size (" << block.x << ", " << block.y << ", " << block.z << ")\n";
			//std::cout << "Grid size (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

			DeviceArray<float> nearest_neighbors;
			nearest_neighbors.create(rows*cols);
			computeCloudResolutionKernel<<<grid,block>>>(rows,cols,vmap,nearest_neighbors.ptr());
			cudaSafeCall (cudaGetLastError ());

			std::vector<float> min_distances;
			nearest_neighbors.download(min_distances);
	
			float mean=0.0f;
			int counter=0;
			for(unsigned int i=0;i<min_distances.size();i++)
			{
				if( min_distances[i] != std::numeric_limits<float>::max() )
				{
					mean+=min_distances[i];
					counter++;
				}
			}

		  	return mean/counter;
		}

		void computeNormalsEigen (const MapArr& vmap, MapArr& nmap)
		{
			//std::cout << "Launching normal estimation kernel...\n";

			int cols = vmap.cols ();
			int rows = vmap.rows () / 3;

			nmap.create (vmap.rows (), vmap.cols ());

			dim3 block (BLOCKDIM_X, BLOCKDIM_Y);
			dim3 grid (1, 1, 1);
			grid.x = divUp (cols, block.x);
			grid.y = divUp (rows, block.y);

			//std::cout << "Block size (" << block.x << ", " << block.y << ", " << block.z << ")\n";
			//std::cout << "Grid size (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

			computeNmapKernelEigen<<<grid, block>>>(rows, cols, vmap, nmap);
			cudaSafeCall (cudaGetLastError ());
			cudaSafeCall (cudaDeviceSynchronize ());
		}

		void bilateralFilter (const DepthMap& src, DepthMap& dst)
		{
			//std::cout << "Launching bilateral filter kernel...\n";

			dst.create( src.rows(), src.cols () );

			dim3 block (BLOCKDIM_X, BLOCKDIM_Y);
			dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

			//std::cout << "Block size (" << block.x << ", " << block.y << ", " << block.z << ")\n";
			//std::cout << "Grid size (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";

			cudaFuncSetCacheConfig (bilateralKernel, cudaFuncCachePreferL1);
			bilateralKernel<<<grid, block>>>(src, dst, 0.5f/ (sigma_space * sigma_space), 0.5f / (sigma_color * sigma_color));

			cudaSafeCall ( cudaGetLastError () );
		}
	}
}
