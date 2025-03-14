from setuptools import setup, find_packages

setup(
    name="noosphere",
    version="0.1.0",
    description="An internet of knowledge graphs with progressive summarization",
    author="Allen Day",
    author_email="allenday@allenday.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "apache-beam[gcp]>=2.49.0,<3.0.0",
        "httpx>=0.27.0,<1.0.0",
        "qdrant-client>=1.7.3,<2.0.0",
        "loguru>=0.7.0,<0.8.0",
        "PyYAML>=6.0.0,<7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "noo-telegram-batch=noosphere.telegram.batch.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)