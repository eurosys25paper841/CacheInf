from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cacheinf',
    version="0.0.1",
    ext_modules=[
        CUDAExtension('cacheinf_cuda', [
            'pybind.cpp',
            'decode_mvs.cpp',
            'decode_mvs_kernel.cu',
            'extract_mvs.cpp',
            'extract_mvs_kernel.cu',
            'fill_activations.cpp',
            'gather_mvs.cpp',
            'pixel_block_utils.cpp',
            'gather_mvs_kernel.cu',
            'tile_recompute_blocks.cpp',
            'tile_recompute_blocks_kernel.cu'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })