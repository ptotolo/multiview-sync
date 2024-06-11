from setuptools import setup, find_packages

setup(
    name='multiview-sync',  # Replace with your package name
    version='0.1.0',  # Start with version 0.1.0 for initial release
    description='Sync videos recorded from different angles and arrange them in a grid',
    author='Pedro Tótolo',
    author_email='ptotolo@ime.usp.br',
    packages=find_packages(exclude=['tests*']),  # Automatically find modules
    install_requires=[  # List any external dependencies here (e.g., 'numpy', 'pandas')
        'math',
        'numpy',
        'subprocess',
        'scipy',
        'os'
    ],
)
