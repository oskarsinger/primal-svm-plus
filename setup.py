from distutils.core import setup

package_name = 'prinfo'

packages = [
    'loaders',
    'servers',
    'models',
    'optimizers',
    'testers'
]

setup(
    name='PrivilegedInformation',
    version='0.01',
    packages=[package_name] + [package_name + '.' + p for p in packages])
