from setuptools import setup, find_packages

setup(
    name="rhd_risk_prediction",
    version="0.1",
    package_dir={"rhd_risk_prediction": "src"},  # Critical change
    packages=["rhd_risk_prediction"],  # Explicit package name
    python_requires=">=3.10",
)