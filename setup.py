from setuptools import setup

setup(name='Boruta',
      version='0.1.3',
      description='Python Implementation of Boruta Feature Selection',
      url='https://github.com/danielhomola/boruta_py',
      download_url='https://github.com/danielhomola/boruta_py/tarball/0.1.3',
      author='Daniel Homola',
      author_email='dani.homola@gmail.com',
      license='BSD 3 clause',
      packages=['boruta'],
      keywords=['feature selection', 'machine learning', 'random forest'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'scipy>=0.17.0'
                        ])
