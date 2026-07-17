# Building and releasing

## Local release checks

```console
python -m pip install -e ".[test,docs,examples]"
ruff check pyradar tests docs/examples
mypy pyradar
python -m pytest --cov=pyradar/base --cov=pyradar/rsp
python -m sphinx -W -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

Inspect both archives before publishing. Neither may contain `archieves`,
`pyradar_sandbox`, tests, generated plots, or demo data.

```console
python -m zipfile -l dist/pyradar-1.0.0rc1-py3-none-any.whl
python -m tarfile -l dist/pyradar-1.0.0rc1.tar.gz
```

Then install the wheel into a clean virtual environment and import it.

## CI and Pages

The CI workflow tests Python 3.10-3.13 on Linux, Windows, and macOS. The Pages
workflow builds Sphinx with warnings treated as errors and deploys the official
Pages artifact. Tagging `v1.0.0rc1` starts the release workflow, which attaches
the wheel, sdist, and SHA256 checksum file as a GitHub prerelease.

The workflow deliberately has no PyPI upload step.
