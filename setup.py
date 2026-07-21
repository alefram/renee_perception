from setuptools import find_packages, setup

package_name = 'renee_perception'

setup(
    name=package_name,
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={'perception': ['configs/*.yaml']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexis',
    maintainer_email='fraumalex@gmail.com',
    description='RENEE perception: offline RGB-D capture -> registered point cloud / mesh tooling.',
    license='TODO: License declaration',
    tests_require=['pytest'],
)
