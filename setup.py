import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(name='omniart_eye_dataset',
                 version='0.1.4',
                 description='A PyTorch dataset of the eyes found in the OmniArt dataset',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/rogierknoester/omniart_eye_dataset',
                 author='Rogier Knoester',
                 author_email='knoesterrogier+omniart@gmail.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 package_data={'omniart_eye_dataset': ['datasets/*.tar.xz.*', 'datasets/*.tar.xz']},
                 install_requires=requirements,
                 zip_safe=False)
