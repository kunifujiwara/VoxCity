# VoxCity Documentation

This directory contains the documentation for VoxCity, built using Sphinx and deployed on Read the Docs.

## Local Development

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Install the package (editable) for API docs
pip install -e .

# Build the documentation
cd docs
make html

# View the documentation
# Open docs/_build/html/index.html in your browser
```

## Read the Docs Deployment

Documentation builds on Read the Docs using `.readthedocs.yml` in the repo root.

### Key settings

- **Python**: 3.12
- **Sphinx config**: `docs/conf.py`
- **Requirements**: `docs/requirements.txt`
- **Output**: HTML

### Customization

- **Theme**: Uses Furo theme (modern, responsive)
- **Logo**: Custom logo in `docs/logo.png`
- **Styling**: Custom CSS in `docs/_static/custom.css`
- **API Documentation**: Auto-generated using sphinx-autoapi

### Troubleshooting

If builds fail:
1. Check the build logs on Read the Docs
2. Ensure all dependencies are in `docs/requirements.txt`
3. Verify the Sphinx configuration in `docs/conf.py`
4. Test locally with `make html` in the docs directory

## Documentation Structure

- `index.md`: Main landing page
- `example.ipynb`: Quick start guide (rendered via myst-nb)
- `examples/`: Tutorial examples
- `autoapi/`: Auto-generated API documentation
- `references.bib`: Bibliography for citations

## Documentation Branch Deployment

A GitHub Actions workflow (`.github/workflows/docs.yml`) can publish the built HTML to the `documentation` branch for GitHub Pages or archival.

### Manual Deployment (optional)

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

## Configuration

The documentation is configured in `conf.py`. Key settings:

- Project name: `voxcity`