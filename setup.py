"""
Setup script for VL-JEPA package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "VL-JEPA: Vision-Language Joint Embedding Predictive Architecture"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "einops>=0.6.1",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "pycocotools>=2.0.6",
    ]

setup(
    name="vl-jepa",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Vision-Language Joint Embedding Predictive Architecture for Jetson Orin Nano",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vl-jepa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vl-jepa-train=train:main",
            "vl-jepa-inference=inference:main",
        ],
    },
)
