from setuptools import setup


requires = [
    "gym==0.10.9",
    "numpy==1.14.5",
    "matplotlib==3.0.2",
    "mpl-finance==0.10.0",
]

setup(name='trading-gym',
      version='0.1.1',
      description='A gym tool for trading.',
      url='https://github.com/mymusise/Trading-Gym',
      author='mymusise',
      author_email='mymusise1@gmail.com',
      license='MIT',
      packages=['trading_gym'],
      install_requires=requires,
      zip_safe=False)
