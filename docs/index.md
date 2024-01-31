# Welcome to Pyles

`Pyles` is a python implementation of the [`Celes`](https://github.com/disordered-photonics/celes) framework (based on [Matlab](https://matlab.mathworks.com/)) and extends its functionality by providing optical parameters, similar to [`MSTM`](https://github.com/dmckwski/MSTM) (based on [Fortran](https://fortran-lang.org/)).

## As of 01.07.2022
The code will be documented as much as possible, to make it easier to use, fix, modify, and/or extend the current code base. Insted of using [Sphinx](https://www.sphinx-doc.org/en/master/), which kinda state of the art [AFAIK](https://www.dictionary.com/browse/afaik), `Pyles` is documented using [MkDocs](https://www.mkdocs.org/), i.e., markdown. To simplify the generation of documentation, the plugin [mkdocstrings](https://mkdocstrings.github.io/) is used to extract the documentation inside the code and present it in `mkdocs`. Currently the [NumpyDoc](https://numpydoc.readthedocs.io/en/latest/format.html) style is used to document the code, which currently prohibts the use of [mkgendocs](https://github.com/davidenunes/mkgendocs), a nice tool to automatically generate a reference page.

If someone has ideas to extend the documentation or its functionality, or objections on what or how something is currently written, please open up an issue so it can be dealt with :)