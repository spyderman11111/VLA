from setuptools import setup

package_name = 'pi0_ros_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pi0_ros_bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shuo Zhang',
    maintainer_email='shuo.zhang@ipk.fraunhofer.de',
    description='ROS 2 bridge to run pi0 model for controlling UR5.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pi0_ros_node = pi0_ros_bridge.main:main',
            'fake_image_publisher = pi0_ros_bridge.fake_image_publisher:main',
        ],
    },
)
