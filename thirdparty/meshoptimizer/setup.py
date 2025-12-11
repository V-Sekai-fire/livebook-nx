"""
Setup script for meshoptimizer Python package.
Builds the C++ shared library during installation via uv/pip.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class BuildLibrary(build_ext):
    """Build the meshoptimizer C++ shared library using CMake"""
    
    def run(self):
        """Build the shared library"""
        if not self.dry_run:
            self.build_library()
        # Call parent to handle Python extensions (if any)
        super().run()
    
    def build_library(self):
        """Build meshoptimizer shared library using CMake"""
        base_dir = Path(__file__).parent.absolute()
        build_dir = base_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Configure with CMake
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
        ]
        
        try:
            subprocess.check_call(
                ["cmake", str(base_dir)] + cmake_args,
                cwd=str(build_dir)
            )
            
            # Build
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=str(build_dir)
            )
            
            # Copy library to package directory so it can be found
            lib_ext = ".so" if sys.platform != "darwin" else ".dylib"
            if sys.platform == "win32":
                lib_ext = ".dll"
            
            # Find the built library
            lib_files = list(build_dir.glob(f"*meshoptimizer*{lib_ext}"))
            if not lib_files:
                # Try Release subdirectory on Windows
                release_dir = build_dir / "Release"
                if release_dir.exists():
                    lib_files = list(release_dir.glob(f"*meshoptimizer*{lib_ext}"))
            
            if lib_files:
                # Copy to package directory
                package_dir = base_dir
                for lib_file in lib_files:
                    dest = package_dir / lib_file.name
                    shutil.copy2(lib_file, dest)
                    print(f"Copied {lib_file.name} to package directory")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not build meshoptimizer library: {e}")
            print("The package will work but will fall back to trimesh if library is not found.")


class BuildPyWithLibrary(build_py):
    """Build Python package and ensure library is built"""
    
    def run(self):
        # Build library first
        build_lib = BuildLibrary(self.distribution)
        build_lib.finalize_options()
        build_lib.run()
        # Then build Python package
        super().run()


setup(
    name="meshoptimizer",
    version="1.0.0",
    description="Python bindings for meshoptimizer - mesh optimization library with screen-space error metrics",
    long_description=open("README_BUILD.md").read() if os.path.exists("README_BUILD.md") else "",
    long_description_content_type="text/markdown",
    author="Arseny Kapoulkine",
    author_email="arseny.kapoulkine@gmail.com",
    url="https://github.com/zeux/meshoptimizer",
    py_modules=["meshoptimizer"],
    python_requires=">=3.8",
    install_requires=["numpy>=1.19.0"],
    setup_requires=["cmake>=3.10"],
    cmdclass={
        "build_ext": BuildLibrary,
        "build_py": BuildPyWithLibrary,
    },
    zip_safe=False,
)
