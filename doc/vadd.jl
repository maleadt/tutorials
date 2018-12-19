#' ---
#' title : GPU Vector addition with Julia
#' author : Tim Besard
#' ---

#' # Introduction

#' In this tutorial, I will demonstrate vector addition on the GPU using a variety of GPU
#' packages. Let's start by defining some data we want to add:

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

#' To upload this data to the GPU, we'll use the CuArrays.jl package

using CuArrays

d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)


#' # Using existing CUDA code

#' One approach is to reuse existing CUDA code that has been written in CUDA C++ and
#' compiled to PTX code with NVCC. For example, let's define a simple vector addition:

vadd_code = """
    extern "C" {

    __global__ void kernel_vadd(const float *a, const float *b, float *c)
    {
        int i = blockIdx.x *blockDim.x + threadIdx.x;
        c[i] = a[i] + b[i];
    }

    }
"""

#' Typically you'd be working with code that has been compiled beforehand, but let's
#' demonstrate how you can robustly compile code using CUDAapi.jl:

using CUDAapi

# write the source to a temporary file
vadd_source = "$(tempname()).cu"
write(vadd_source, vadd_code)

# build a PTX object
vadd_object = "$(tempname()).ptx"

# we'll be compiling for the current device
import CUDAdrv: device, capability
dev = device()
cap = capability(dev)

toolkit = CUDAapi.find_toolkit()
nvcc = CUDAapi.find_cuda_binary("nvcc", toolkit)
toolchain = CUDAapi.find_toolchain(toolkit)
flags = `-ccbin=$(toolchain.host_compiler) -arch=sm_$(cap.major)$(cap.minor)`
run(`$nvcc $flags -ptx -o $(vadd_object) $(vadd_source)`)


#' To load the resulting PTX object, and perform other low-level CUDA operations, we use the
#' CUDAdrv.jl package that wraps the CUDA driver library. We also load the specific CUDA
#' function we'll be calling, `kernel_vadd`.

using CUDAdrv

md = CuModuleFile(vadd_object)
kernel = CuFunction(md, "kernel_vadd")

#' Now we can call the vector addition function by using `cudacall`, a low-level function
#' that mimics `ccall`. We launch a number of threads that corresponds with the number of
#' elements in our array:

len = prod(dims)
cudacall(kernel, Tuple{Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}}, d_a, d_b, d_c; threads=len)

#' Let's now verify the addition has succeeded:

using Test

@test a+b ≈ Array(d_c)


#' # Using CUDAnative kernels

using CUDAnative

function vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end
@cuda threads=len vadd(d_a, d_b, d_c)

@test a+b ≈ Array(d_c)


#' # Using CuArrays kernels

@test a+b ≈ Array(d_a .+ d_b)
