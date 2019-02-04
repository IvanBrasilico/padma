from setuptools import find_packages, setup

setup(
    name='AJNA_MOD-padma',
    description='Visao computacional e Aprendizado de Maquina na Vigilancia Aduaneira',
    version='0.0.1',
    url='https://github.com/IvanBrasilico/padma',
    license='GPL',
    author='Ivan Brasilico',
    author_email='brasilico.ivan@gmail.com',
    packages=find_packages(),
    install_requires=[
        'Flask',
        'flask_bootstrap',
        'flask_login',
        'flask_nav',
        'flask_wtf',
        'gevent',
        'gunicorn',
        'keras',
        'h5py',
        'imutils',
        'matplotlib',
        'numpy',
        'opencv-python',
        'Pillow',
        'pymongo',
        'raven',
        'redis',
        'scipy',
        'scikit-learn==0.19.1',
        'tensorflow==1.5.0',
    ],
    dependency_links=[
        'git+https://github.com/broadinstitute/keras-resnet',
        'git+https://github.com/IvanBrasilico/keras-retinanet.git'
    ],
    tests_require=['pytest'],
    test_suite="tests",
    package_data={
    },
    extras_require={
        'dev': [
            'bandit',
            'codecov',
            'coverage',
            'flake8',
            'flake8-quotes',
            'flask-webtest',
            'isort',
            'ipykernel',
            'pandas',
            'pytest',
            'pytest-cov',
            'pytest-mock',
            'requests',
            'testfixtures',
            'tox'
        ],
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Software Development :: User Interfaces',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3.5',
    ],
)
