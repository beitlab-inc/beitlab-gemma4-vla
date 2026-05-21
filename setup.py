from setuptools import setup, find_packages

setup(
    name="gemma4_vla",
    version="0.1.0",
    description="Vision-Language-Action model based on Gemma 4 (pi0-style architecture)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gemma4VLA contributors",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "peft>=0.11.0",
        "h5py>=3.10.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "huggingface_hub>=0.23.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0"],
        "quant": ["bitsandbytes>=0.43.0"],
        "logging": ["wandb>=0.17.0"],
        "metaworld": [
            "gymnasium>=1.0.0",
            "mujoco>=3.0.0",
            "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@master",
            "imageio>=2.30.0",
            "opencv-python>=4.8.0",
        ],
        "observability": [
            "rerun-sdk>=0.22.0",
            "mlflow>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemma4vla-train=gemma4_vla.train:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
