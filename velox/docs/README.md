# Velox Documentation

We use Sphinx to generate documentation and GitHub pages to publish it.
- Sphinx: https://pythonhosted.org/an_example_pypi_project/sphinx.html
- GitHub pages: https://pages.github.com/

## Building

To install Sphinx: `easy_install -U sphinx`

`sphinx-quickstart` command was used to generate the initial Makefile and config.

### Building PyVelox Components

If you're using a conda environment, it can be easily installed by `conda install` command.

Pandoc is also used to generate `.rst` files from existing markdown files. Refer to installation
instructions [here](https://pandoc.org/installing.html).

To build the documentation, e.g. generate HTML files from .rst files:

Run the `./scripts/gen-docs.sh` script from the base directory.

Navigate to
`velox/docs/_build/html/index.html` in your browser to view the documentation.

## Publishing

GitHub pages is configured to display the contents of the top-level docs directory
found in the gh-pages branch. The documentation is available at
https://facebookincubator.github.io/velox.

To publish updated documentation, copy the contents of the _build/html
directory to the top-level docs folder and push to gh-pages branch.

```
# Make sure 'main' is updated to the top of the tree.
# Make a new branch.
git checkout -b update-docs main

# Generate the documentation.
./scripts/gen-docs.sh

# Copy documentation files to the top-level docs folder.
cp -R _build/html/* ../../docs

# Commit the changes.
git add ../../
git commit -m "Update documentation"

# Update gh-pages branch in the upstream.
# This will get the website updated.
git push -f upstream update-docs:gh-pages
```
