from setuptools import setup


requires = [
    "gym",
    "numpy",
    "pandas",
    "matplotlib",
    "mpl-finance",
]

setup(name='trading-gym',
      version='0.1.8',
      description='A gym tool for trading.',
      url='https://github.com/mymusise/Trading-Gym',
      author='mymusise',
      author_email='mymusise1@gmail.com',
      license='MIT',
      packages=['trading_gym'],
      install_requires=requires,
      zip_safe=False)
