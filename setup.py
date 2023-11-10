import setuptools

REQUIRED_PACKAGES = [
    "numpy==1.23.5",
    "pandas==1.3.5",
    "scipy==1.9.1",
    "empyrical",
    "pyfolio",
    "pandas-ta==0.3.14b0",
    "TA-Lib",
    "yfinance",
]

PACKAGE_NAME = "stock_env"
PACKAGE_VERSION = "1.1.0"

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description="Stock Env",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
)
