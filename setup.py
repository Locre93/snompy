from distutils.core import setup

setup(name='snompy',
      version='0.1.0',
      description='Package for analyzing NeaSpec s-SNOM scan',
      authors=['Lorenzo Orsini','Matteo Ceccanti'],
      author_email=['lorenzo.orsini@icfo.eu','matteo.caccanti@icfo.eu'],
      url='https://github.com/Locre93/snompy',
      install_requires=['numpy','matplotlib','scipy','pandas'])