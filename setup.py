from setuptools import setup, find_packages
setup(
    name='difflearn',  # 包的名称
    packages=find_packages(),  # 包含的模块
    version='1.0.2',
    author='Jiacheng Leng',
    author_email='jcleng@amss.ac.cn',
    description='Some useful tools for differential network inference with python.',
    long_description=open('README.md').read(),
    license='LICENSE.txt',
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
        'numpy',
        'matplotlib',
        'scipy',
        'rpy2',
        'networkx',
        'joblib',
        'progressbar2'
    ],
    long_description_content_type="text/markdown"
)
