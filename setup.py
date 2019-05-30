import torch
from setuptools import setup, find_packages

import sys

if not torch.cuda.is_available():
    print("Warning: Torch did not find available GPUs on this system.\n",
          "If your intention is to cross-compile, this is not an error.")

print("torch.__version__  = ", torch.__version__)
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
      raise RuntimeError("Apex requires Pytorch 0.4 or newer.\n" +
                         "The latest stable release can be obtained from https://pytorch.org/")

cmdclass = {}
ext_modules = []

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError("--cpp_ext requires Pytorch 1.0 or later, "
                           "found torch.__version__ = {}".format(torch.__version))
    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension

if "--cpp_ext" in sys.argv:
    from torch.utils.cpp_extension import CppExtension
    sys.argv.remove("--cpp_ext")
    ext_modules.append(
        CppExtension('apex_C',
                     ['csrc/flatten_unflatten.cpp',]))

if "--cuda_ext" in sys.argv:
    from torch.utils.cpp_extension import CUDAExtension
    sys.argv.remove("--cuda_ext")

    if torch.utils.cpp_extension.CUDA_HOME is None:
        raise RuntimeError("--cuda_ext was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
    else:
        # Set up macros for forward/backward compatibility hack around
        # https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']

        ext_modules.append(
            CUDAExtension(name='amp_C',
                          sources=['csrc/amp_C_frontend.cpp',
                                   'csrc/multi_tensor_scale_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3'],
                                              'nvcc':['-lineinfo',
                                                      '-O3',
                                                      '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='fused_adam_cuda',
                          sources=['csrc/fused_adam_cuda.cpp',
                                   'csrc/fused_adam_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-O3', 
                                                      '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='com_fused_adam_cuda',
                          sources=['csrc/com_fused_adam_cuda.cpp',
                                   'csrc/com_fused_adam_cuda_kernel.cu'], 
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-O3',
                                              '--use_fast_math']}))
        ext_modules.append(
            CUDAExtension(name='fused_lamb_cuda',
                          sources=['csrc/fused_lamb_cuda.cpp',
                                   'csrc/fused_lamb_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3',],
                                              'nvcc':['-O3', 
                                                      '--use_fast_math',
                                                      '-I cuda_inc/']}))
        
        ext_modules.append(
            CUDAExtension(name='syncbn',
                          sources=['csrc/syncbn.cpp',
                                   'csrc/welford.cu']))
        ext_modules.append(
            CUDAExtension(name='fused_layer_norm_cuda',
                          sources=['csrc/layer_norm_cuda.cpp',
                                   'csrc/layer_norm_cuda_kernel.cu'],
                          extra_compile_args={'cxx': ['-O3'] + version_ge_1_1,
                                              'nvcc':['-maxrregcount=50',
                                                      '-O3', 
                                                      '--use_fast_math'] + version_ge_1_1}))

setup(
    name='apex',
    version='0.1',
    packages=find_packages(exclude=('build',
                                    'csrc',
                                    'include',
                                    'tests',
                                    'dist',
                                    'docs',
                                    'tests',
                                    'examples',
                                    'apex.egg-info',)),
    description='PyTorch Extensions written by NVIDIA',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
