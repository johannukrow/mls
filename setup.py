from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="mls",
        sources=["mls.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="mls",
    version="0.1.0",
    description="A weighted similarity interpolation package implemented in Cython.",
    ext_modules=cythonize(extensions),
)
