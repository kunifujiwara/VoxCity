# VoxCity Documentation

This directory contains the documentation for VoxCity, built using Sphinx and deployed on Read the Docs.

## Local Development

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# View the documentation
# Open docs/_build/html/index.html in your browser
```

## Read the Docs Deployment

The documentation is automatically deployed to Read the Docs when changes are pushed to the main branch.

### Setup Instructions

1. **Connect to Read the Docs**:
   - Go to [readthedocs.org](https://readthedocs.org)
   - Sign in with your GitHub account
   - Click "Import a Project"
   - Select your VoxCity repository

2. **Configuration**:
   - The `.readthedocs.yml` file in the root directory configures the build
   - Documentation will be built from the `docs/` directory
   - The build uses Python 3.11 and Ubuntu 22.04

3. **Automatic Builds**:
   - Documentation builds automatically on every push to main
   - New versions are created for git tags
   - Build status is shown in the GitHub repository

### Configuration Details

The `.readthedocs.yml` file specifies:
- **Python version**: 3.11
- **Build system**: Sphinx
- **Configuration file**: `docs/conf.py`
- **Requirements**: `docs/requirements.txt`
- **Output formats**: HTML, PDF, ePub

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

## Configuration

The documentation is configured in `conf.py`. Key settings:

- Project name: `voxcity`