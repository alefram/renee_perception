from setuptools import find_packages, setup

package_name = 'renception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(where='.'),
    package_data={'renception': ['configs/*.yaml']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexis',
    maintainer_email='fraumalex@gmail.com',
    description='RENEE perception: offline RGB-D capture -> registered point cloud / mesh tooling.',
    license='TODO: License declaration',
    tests_require=['pytest'],
)
