from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pcpr',
    version="0.1",
    ext_modules=[
        CUDAExtension('pcpr', [
            'CloudProjection/pcpr_cuda.cpp',
            'CloudProjection/point_render.cu'
        ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })