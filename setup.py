from scipy_distutils.core import setup, Extension

lsodar = Extension(name = 'SloppyCell._lsodar',
                   sources = ['lsodar.pyf', 'odepack/opkdmain.f', 
                              'odepack/opkda1.f', 'odepack/opkda2.f'])

setup(name='SloppyCell',
      version='0.2',
      author='Ryan Gutenkunst',
      author_email='rng7@cornell.edu',
      url='http://sloppycell.sourceforge.net',
      packages=['SloppyCell', 
                'SloppyCell.ReactionNetworks', 
                'SloppyCell.Testing'
                ],
      package_dir={'SloppyCell': ''},

      ext_modules = [lsodar]
      )
