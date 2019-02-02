from setuptools import setup


requires = ["stable-baselines==2.4.0", "mpl-finance==0.10.0"]

setup(name='trading-gym',
      version='0.1.1',
      description='A gym tool for trading.',
      url='https://github.com/mymusise/Trading-Gym',
      author='mymusise',
      author_email='mymusise1@gmail.com',
      license='MIT',
      packages=['trading_gym'],
      zip_safe=False)
