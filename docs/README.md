# Building Documentation

## Install pre-requisites

Within the same conda environment created to build `pyvelox` install the
following conda packages

```bash
conda install -c conda-forge pandoc
conda install -c anaconda sphinx
conda install -c conda-forge doxygen
```

## Building Doxygen documentation

```bash
pushd docs/doxygen
doxygen
popd
```

## Building Sphinx documentation

```bash
pushd docs
pandoc ../pyvelox/README.md --from markdown --to rst -s -o sphinx/source/README_generated_pyvelox.rst
pandoc README.md --from markdown --to rst -s -o sphinx/source/README_generated_docs.rst
pushd sphinx
sphinx-build source _build/html
make html
```

Generated `html` files are at `docs/sphinx/_build`.


