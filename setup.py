from setuptools import setup
with open("README.md") as fh:
    long_description = fh.read()
    
if __name__ == '__main__':
    setup(name = 'apdt',
          version = '0.1',
          description = 'apdt',
          author = 'Zhiyuan-Wu',
          author_email = '86562713@qq.com',
          url = 'https://github.com/Zhiyuan-Wu/apdt',
          maintainer = 'Zhiyuan-Wu',
          maintainer_email = '86562713@qq.com',
          long_description = long_description,
          long_description_content_type="text/markdown",          
          install_requires = ['numpy', 'pandas', 'sklearn'],
          license = 'Apache License Version 2.0',
          py_modules = ['apdt'],
          classifiers = (
              "Development Status :: 4 - Beta",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.6",
              "Operating System :: OS Independent",
          ),
          )
