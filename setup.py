import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(name='omniart_eye_dataset',
                 version='0.1.1',
                 description='A PyTorch dataset of the eyes found in the OmniArt dataset',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/rogierknoester/omniart_eye_dataset',
                 author='Rogier Knoester',
                 author_email='knoesterrogier+omniart@gmail.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
