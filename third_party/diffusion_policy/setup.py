from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent
DESCRIPTION = "Diffusion policies for Nero contact-rich robotic manipulation."
README_PATH = ROOT / "README.md"
LONG_DESCRIPTION = (
    README_PATH.read_text(encoding="utf-8")
    if README_PATH.is_file()
    else DESCRIPTION
)

CORE_DEPENDENCIES = [
    # Keep shared ABI-sensitive packages aligned with nero_ws.
    "numpy>=2.2,<2.3",
    "h5py>=3.11",
    "scipy>=1.10,<1.16",
    "opencv-python-headless>=4.9,<4.13",
    "PyYAML>=6.0",
    "matplotlib>=3.7",
    "mujoco>=3.3,<4",
    "osqp>=1,<2",
    "pin>=3,<4; platform_system == 'Linux'",
    "rerun-sdk>=0.26,<0.27",
    "cmeel-tinyxml2>=10,<11; platform_system == 'Linux'",
    "cmeel-urdfdom>=4,<5; platform_system == 'Linux'",
    "numba>=0.66,<0.67",
    # Force-aware Diffusion Transformer runtime.
    "torch>=2.3,<3",
    "torchvision>=0.18,<1",
    "hydra-core>=1.3,<1.4",
    "omegaconf>=2.3,<2.4",
    "einops>=0.8,<0.9",
    "diffusers>=0.35,<0.36",
    "transformers>=4.56,<5",
    "huggingface-hub>=0.35,<1",
    "zarr>=2.18,<3",
    "numcodecs>=0.13,<0.14",
    "pyarrow>=20,<21",
    "tqdm>=4.67,<5",
    "Pillow>=11,<13",
    "filelock>=3.16,<4",
]

EXTRAS = {
    "lerobot": [
        "lerobot==0.4.0",
    ],
    "training": [
        "accelerate>=1.10,<2",
        "wandb>=0.21,<0.23",
        "tensorboard>=2.15,<3",
        "dill>=0.3.8,<0.4",
    ],
    "test": [
        "pytest>=7,<9",
    ],
}
EXTRAS["all"] = sorted(
    {dependency for dependencies in EXTRAS.values() for dependency in dependencies}
)


setup(
    name="diffusion-policy",
    version="0.2.0",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Diffusion Policy Team",
    license="MIT",
    python_requires=">=3.10,<3.11",
    packages=find_namespace_packages(include=["diffusion_policy*"]),
    include_package_data=True,
    package_data={
        "diffusion_policy": [
            "config/**/*.yaml",
            "config/**/*.yml",
        ]
    },
    install_requires=CORE_DEPENDENCIES,
    extras_require=EXTRAS,
    zip_safe=False,
)
