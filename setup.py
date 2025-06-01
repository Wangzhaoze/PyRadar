from setuptools import setup, find_packages

setup(
    name="pyradar",
    version="0.1.0",
    description="A Python radar signal processing library with optional C++ acceleration",
    author="Zhaoze Wang",
    author_email="wangzhaoze@outlook.com",
    license="MIT",
    packages=["pyradar"],  # 自动找到 pyradar 子目录作为包
    include_package_data=True,
    package_data={"pyradar": ["*.so", "*.pyd"]},  # 如果你有 C/C++ 编译结果
    install_requires=[
        "matplotlib",
        "numpy",
        "open3d",
        "pillow",
        "pre-commit",
        "pyparsing",
        "pyyaml",
        "scipy",
        "tabulate"
    ],
    python_requires=">=3.9",
    url="https://github.com/Wangzhaoze/pyradar",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
