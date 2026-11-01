from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Flags aggiuntivi per MSVC e GCC/Clang
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/std:c++20", "/O2", "/fp:precise"]
else:
    extra_compile_args = ["-std=c++20", "-O3", "-fPIC"]

ext_modules = [
    Pybind11Extension(
        "py_core_dsp",
        ["cpp/bindings/py_core_dsp.cpp"],
        include_dirs=["cpp/core_dsp/include"],
        cxx_std=20,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="ddsp_mastering",
    version="0.1.0",
    author="Giovanni - PoliMi Music & Acoustic Engineering",
    description="White-Box Differentiable DSP AutoMastering Twin Engine",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.11",
)
