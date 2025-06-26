# VoxCity Documentation

This directory contains the documentation for the VoxCity project, built using Sphinx.

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Building Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the documentation:**
   ```bash
   make html
   ```

3. **View the documentation:**
   Open `_build/html/index.html` in your web browser.

### Using the build script

Alternatively, you can use the provided build script:

```bash
python build_docs.py
```

## Documentation Branch Deployment

The documentation is automatically deployed to the `documentation` branch using GitHub Actions. The workflow is defined in `.github/workflows/docs.yml`.

### Manual Deployment

If you need to deploy manually:

1. Build the documentation:
   ```bash
   make html
   ```

2. The built files will be in `_build/html/`

3. Deploy to the documentation branch:
   ```bash
   cd docs/_build/html
   git init
   git add -A .
   git commit -m "Update documentation"
   git push -f origin HEAD:documentation
   ```

## Documentation Structure

- `index.md` - Main documentation page (includes README.md)
- `examples/` - Tutorial and example notebooks
- `autoapi/` - Auto-generated API documentation
- `_static/` - Static assets (CSS, images, etc.)
- `_build/` - Built documentation (generated)

## Configuration

The documentation is configured in `conf.py`. Key settings:

- Project name: `voxcity`
- Theme: `furo`
- Extensions: `myst-nb`, `autoapi`, `sphinxcontrib-bibtex`
- AutoAPI: Generates documentation from source code in `../src/`

## Adding Content

1. **New pages:** Add `.md` or `.rst` files and include them in the appropriate toctree
2. **Examples:** Add Jupyter notebooks to the `examples/` directory
3. **API docs:** Auto-generated from docstrings in the source code
4. **References:** Update `references.bib` and `references.md`

## Troubleshooting

- **Build errors:** Check that all dependencies are installed
- **Missing modules:** Ensure the voxcity package is installed in editable mode (`pip install -e ..`)
- **AutoAPI issues:** Check that the source code path in `conf.py` is correct 