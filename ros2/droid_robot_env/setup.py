from setuptools import setup

package_name = "droid_robot_env"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "lerobot>=0.4.4; python_version >= '3.10'"],
    zip_safe=True,
    maintainer="DROID Maintainer",
    maintainer_email="todo@example.com",
    description="ROS2 wrapper nodes for DROID RobotEnv.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "franka_robot_env_node = droid_robot_env.franka_robot_env_node:main",
            "lerobot_dataset_collector_node = droid_robot_env.lerobot_dataset_collector_node:main",
        ],
    },
)
