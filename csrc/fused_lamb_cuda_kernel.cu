#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
#include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>

#include <iostream>

#include <helper_functions.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "type_shim.h"

typedef enum{
    ADAM_MODE_0   =0, // eps under square root
    ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

static int sm_count=-1;
static int num_blocks_per_sm=-1;

//s_a and s_b are in shared memory
//g_a and g_b are in shared memory
template <class T, int blockSize>
__device__ void
reduce_block_in_shared_memory(T *s_a, T *s_b, T* g_a, T* g_b)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // perform block reduction in shared memory,
    unsigned int tid = cta.thread_rank();
    
    T a_sum = s_a[tid];
    T b_sum = s_b[tid];

    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 256];
        s_b[tid] = b_sum = b_sum + s_b[tid + 256];
    
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 128];
        s_b[tid] = b_sum = b_sum + s_b[tid + 128];
    
    }

    cg::sync(cta);
    
    if ((blockSize >= 128) && (tid < 64))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 64];
        s_b[tid] = b_sum = b_sum + s_b[tid + 64];
    
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) 
        {
            a_sum = a_sum + s_a[tid + 32];
            b_sum = b_sum + s_b[tid + 32];
        }

        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
             a_sum += active.shfl_down(a_sum, offset);
             b_sum += active.shfl_down(a_sum, offset);
        
        }
    }
#else
    if ((blockSize >= 64) && (tid < 32))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 32];
        s_b[tid] = b_sum = b_sum + s_b[tid + 32];
    
    }
    
    cg::sync(cta);

    if ((blockSize >= 32) && (tid < 16))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 16];
        s_b[tid] = b_sum = b_sum + s_b[tid + 16];
    
    }
    
    cg::sync(cta);

    if ((blockSize >= 16) && (tid < 8))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 8];
        s_b[tid] = b_sum = b_sum + s_b[tid + 8];
    
    }
    
    cg::sync(cta);

    if ((blockSize >= 8) && (tid < 4))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 4];
        s_b[tid] = b_sum = b_sum + s_b[tid + 4];
    
    }
    
    cg::sync(cta);

    if ((blockSize >= 4) && (tid < 2))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 2];
        s_b[tid] = b_sum = b_sum + s_b[tid + 2];
    
    }
    
    cg::sync(cta);

    if ((blockSize >= 2) && (tid < 1))
    {
        s_a[tid] = a_sum = a_sum + s_a[tid + 1];
        s_b[tid] = b_sum = b_sum + s_b[tid + 1];
    
    }
    
    cg::sync(cta);

#endif

    // write result for this block to global mem
    if (tid == 0){
        g_a[blockIdx.x] = a_sum;
        g_b[blockIdx.x] = b_sum;
    } 
}

template <typename T, int blockSize>
__device__ void reduce_two_vectors_in_register(T a, T b, T* g_a, T* g_b, cg::grid_group &cgg){
 
    const int threadIdInBlock = cg::this_thread_block().thread_rank();

    extern __shared__ float s_a[];
    extern __shared__ float s_b[];

    s_a[threadIdInBlock] = a;
    s_b[threadIdInBlock] = b;

    reduce_block_in_shared_memory<T,blockSize>(s_a, s_b ,g_a, g_b);

    cg::sync(cgg);

    if (blockId == 0){
        s_a[threadIdInBlock] = g_a[threadIdInBlock];
        s_b[threadIdInBlock] = g_b[threadIdInBlock];

        if threadIdInBlock > cg::this_grid().size()
            s_a[threadIdInBlock] = 0.0;
            s_b[threadIdInBlock] = 0.0

        reduce_block_in_shared_memory<T,blockSize>(s_a, s_b, g_a, g_b);
    }
    cg::sync(cgg);

}

template <typename T, typename GRAD_T>
__global__ void lamb_cuda_kernel(
        T* __restrict__ p,
        GRAD_T* __restrict__ p_copy, // For mixed precision training, pass NULL if not needed
        T* __restrict__ m,
        T* __restrict__ v,
        const GRAD_T * __restrict__ g,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        const size_t tsize,
        adamMode_t mode,
        const float decay,
        T* __restrict__ w_l2_i,
        T* __restrict__ u_l2_i)
{
        cg::grid_group cgg = cg::this_grid();

        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;
        

        float reg_w = 0;
        float reg_u = 0;
        
        for (int j = i; j < tsize; j+=totThreads) {
                T scaled_grad = g[j]/grad_scale;
                float pj = (float)p[j];
                m[j] = b1*m[j] + (1-b1)*scaled_grad;
                v[j] = b2*v[j] + (1-b2)*scaled_grad*scaled_grad;
                float denom;
                if (mode == ADAM_MODE_0)
                    denom = sqrtf(v[j] + eps);
                else // Mode 1
                    denom = sqrtf(v[j]) + eps;
                float update = (m[j]/denom) + (decay*p[j]);
                
                reg_u += update * update;
                reg_w += pj * pj; 
                
        }

        reduce_two_vectors_in_register<T,512>(reg_w, reg_u, w_l2_i, u_l2_i, cgg);
        
        reg_w = sqrtf(w_l2_i[0]);
        reg_u = sqrtf(u_l2_i[0]);

        float lamb_coeff = 1.0;

        if (reg_w !=0 and reg_u !=0)
            lamb_coeff = reg_w/reg_u;
    
        for (int j = i; j < tsize; j+=totThreads) {
            float pj = (float)p[j];
            float mj = m[j];
            float vj = v[j];
            float denom;
            if (mode == ADAM_MODE_0)
                denom = sqrtf(vj + eps);
            else // Mode 1
                denom = sqrtf(vj) + eps;
            float update = (mj/denom) + (decay*pj);
            
            pj = pj - (step_size * lamb_coeff * update);
            p[j] = pj;
            if (p_copy != NULL) p_copy[j] = (GRAD_T) pj;
    }
}

void fused_lamb_cuda(
        at::Tensor & p,
        at::Tensor & p_copy,
        at::Tensor & m,
        at::Tensor & v,
        at::Tensor & g,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float grad_scale,
        int step,
        int mode,
        int bias_correction,
        float decay,
        at::Tensor & w_l2_i,
        at::Tensor & u_l2_i)
{
//        using namespace at;

        //Get tensor size
        int tsize = p.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        int num_blocks = (tsize+threadsPerBlock-1)/threadsPerBlock;
        if (num_blocks > 512) num_blocks=512;
        int smemsize = 2 * threadsPerBlock * sizeof(float);
        
        if (sm_count == -1){
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop,0);
            sm_count = prop.multiProcessorCount;
            
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, lamb_cuda_kernel<float,float>, threadsPerBlock, smemsize );
        }

        int max_active_blocks = num_blocks_per_sm * sm_count;
        if (num_blocks > max_active_blocks) num_blocks = max_active_blocks;
        std::cout<<"Num Blocks, Num Threads "<<num_blocks<<", "<<threadsPerBlock<<std::endl;

        const dim3 blocks(num_blocks);
        const dim3 threads(threadsPerBlock);

        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
        //Constants
        float step_size = 0;
        if (bias_correction == 1) {
            const float bias_correction1 = 1 - std::pow(beta1, step);
            const float bias_correction2 = 1 - std::pow(beta2, step);
            step_size = lr * std::sqrt(bias_correction2)/bias_correction1;
        }
        else {
            step_size = lr;
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        float lbeta1 = beta1;
        float lbeta2 = beta2;
        float leps = eps;
        float lgrad_scale = grad_scale;
        int lmode = mode;
        float ldecay=decay;

        if (g.type().scalarType() == at::ScalarType::Half) {
//all other values should be fp32 for half gradients
            AT_ASSERTM(p.type().scalarType() == at::ScalarType::Float, "expected parameter to be of float type");
//dispatch is done on the gradient type
            using namespace at; // prevents "toString is undefined" errors
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(TypeShim(g.type()), "lamb_cuda_kernel", ([&] {
                using accscalar_t = at::acc_type<scalar_t, true>;

            


                void *kernelArgs[] ={
                        (void*)p.data<accscalar_t>(),
                        (void*)(p_copy.numel() ? p_copy.data<scalar_t>() : NULL),
                        (void*)m.data<accscalar_t>(),
                        (void*)v.data<accscalar_t>(),
                        (void*)g.data<scalar_t>(),
                        (void*)&lbeta1,
                        (void*)&lbeta2,
                        (void*)&leps,
                        (void*)&lgrad_scale,
                        (void*)&step_size,
                        (void*)&tsize,
                        (void*)&lmode,
                        (void*)&ldecay,
                        (void*)w_l2_i.data<accscalar_t>(),
                        (void*)u_l2_i.data<accscalar_t>()

                };

                cudaLaunchCooperativeKernel((void*)lamb_cuda_kernel<accscalar_t, scalar_t>, blocks, threads, kernelArgs, smemsize, stream);
                /*lamb_cuda_kernel<accscalar_t, scalar_t><<<blocks,threadsPerBlock, smemsize, stream>>>(
                        p.data<accscalar_t>(),
                        p_copy.numel() ? p_copy.data<scalar_t>() : NULL,
                        m.data<accscalar_t>(),
                        v.data<accscalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay,
                        w_l2_i.data<accscalar_t>(),
                        u_l2_i.data<accscalar_t>());*/
            }));
      } else {
            using namespace at;
            AT_DISPATCH_FLOATING_TYPES(TypeShim(g.type()), "lamb_cuda_kernel", ([&] {

                scalar_t* nullptr1 = NULL;
                void *kernelArgs[] ={
                    (void*)p.data<scalar_t>(),
                    (void*)&nullptr1,
                    (void*)m.data<scalar_t>(),
                    (void*)v.data<scalar_t>(),
                    (void*)g.data<scalar_t>(),
                    (void*)&lbeta1,
                    (void*)&lbeta2,
                    (void*)&leps,
                    (void*)&lgrad_scale,
                    (void*)&step_size,
                    (void*)&tsize,
                    (void*)&lmode,
                    (void*)&ldecay,
                    (void*)w_l2_i.data<scalar_t>(),
                    (void*)u_l2_i.data<scalar_t>()

                };
                cudaLaunchCooperativeKernel((void*)lamb_cuda_kernel<scalar_t, scalar_t>, blocks, threads, kernelArgs, smemsize, stream);
                /*lamb_cuda_kernel<scalar_t, scalar_t><<<blocks,threadsPerBlock, smemsize, stream>>>(
                        p.data<scalar_t>(),
                        NULL, //don't output p_copy for fp32, it's wasted write
                        m.data<scalar_t>(),
                        v.data<scalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay,
                        w_l2_i.data<accscalar_t>(),
                        u_l2_i.data<accscalar_t>());*/
            }));
      }
      THCudaCheck(cudaGetLastError());

}

//template __device__ void reduce_two_vectors_in_register<float,512>(float a, float b, float* g_a, float* g_b, cg::grid_group &cgg);