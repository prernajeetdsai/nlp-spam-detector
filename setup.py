from setuptools import setup, find_packages

setup(
    name="nlp-spam-detector",
    version="0.1.0",
    author="Prerna",
    description="NLP Spam Classifier with SHAP Explainability",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "flask==3.0.3",
        "scikit-learn==1.5.1",
        "shap==0.46.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "gunicorn==22.0.0",
    ],
)
