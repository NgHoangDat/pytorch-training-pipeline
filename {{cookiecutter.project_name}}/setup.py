import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    requirements = fh.readlines()

__VERSION__ = "{{cookiecutter.version}}"
__DESCRIPTION__ = "{{cookiecutter.description}}"

entry_points = (
    "{{cookiecutter.package_name}} = {{cookiecutter.package_name}}.__main__:main",
)


setuptools.setup(
    name="{{cookiecutter.package_name}}",
    packages=setuptools.find_packages(),
    version=__VERSION__,
    author="{{cookiecutter.name}}",
    author_email="{{cookiecutter.email}}",
    description=__DESCRIPTION__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": entry_points
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.8',
    install_requires=requirements,
    include_package_data=True,
    keywords=[],
)
