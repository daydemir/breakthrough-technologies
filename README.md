# Senseable Python Template

This template is a starting point for researchers who want to publish the code they have written for their research. 

There are essentially three ways to publish your code so it can be reused (and potentially cited):

1. Publish the code as is
2. Publish a Python package to PyPI
3. Publish as peer-reviewed software (to [pyOpenSci](https://www.pyopensci.org) or the [Journal of Open Source Software](https://joss.theoj.org))

If you hope to do either (2) or (3), this template provides some guidance on how to structure your code for package deployment and for peer review. If you only plan to do (1), this can still be a useful template to get started with.

## Getting Started

### Fork this repo

### Setup Poetry

In order to maintain clean virtual environments, we recommend managing your dependencies using [Poetry](https://python-poetry.org). This will make it so you only need one tool to cover the following: managing dependencies, creating local environments, and publishing your package to PyPI.

#### Install Poetry 

Simple installation with `pipx`
```
pipx install poetry
```

See detailed instructions [here](https://python-poetry.org/docs/#installation).

#### Configure in-project virtual environments
```
poetry config virtualenvs.in-project true
```

#### Run install to create virtual environment
```
poetry install
```


## Publishing Your Code
* https://github.com/microsoft/python-package-template/blob/main/README.md
* https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
* https://github.com/kennethreitz/setup.py?tab=readme-ov-file
* https://github.com/navdeep-G/setup.py
* https://github.com/navdeep-G/samplemod?tab=readme-ov-file
* https://figshare.com
* https://joss.theoj.org
* https://www.pyopensci.org
* https://github.com/audreyfeldroy/cookiecutter-pypackage
* https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html
* https://docs.python-guide.org/writing/structure/

### Prepare for Publishing

Update the following content:
* Update year and name in LICENSE
* Review and update CODE_OF_CONDUCT as desired. This is prefilled using the language provided by the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) as recommended by the [pyOpenSci documentation](https://www.pyopensci.org/python-package-guide/documentation/repository-files/code-of-conduct-file.html#why-you-need-a-code-of-conduct).
* Note the CHANGELOG, and plan update it with new releases. More details on good practices at [pyOpenSci](https://www.pyopensci.org/python-package-guide/documentation/repository-files/changelog-file.html#what-does-it-look-like).
