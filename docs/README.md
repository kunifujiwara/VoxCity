# VoxCity Documentation

This directory contains the documentation for VoxCity, a Python package for 3D voxel city model generation and urban simulation.

## Building Documentation

### Prerequisites

- Python 3.12+
- Required packages (install with `pip install -r requirements.txt`)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build documentation:**
   ```bash
   # Using make (Linux/Mac)
   make html
   
   # Using make.bat (Windows)
   make.bat html
   
   # Using Python script
   python build_docs.py
   ```

3. **Serve locally (optional):**
   ```bash
   python build_docs.py --serve
   ```

### Build Options

- `make html` - Build HTML documentation
- `make clean` - Clean build directory
- `make help` - Show all available commands
- `python build_docs.py --clean --serve` - Clean build and serve locally

## Documentation Structure

```
docs/
├── index.md              # Main documentation page
├── examples/             # Tutorial notebooks
│   ├── demo_basic.ipynb
│   ├── demo_solar_analysis.ipynb
│   ├── demo_view_index.ipynb
│   ├── demo_landmark_visibility.ipynb
│   ├── demo_network_analysis.ipynb
│   └── demo_export_formats.ipynb
├── references.md         # References and citations
├── references.bib        # Bibliography file
├── changelog.md          # Version history
├── contributing.md       # Contribution guidelines
├── conduct.md           # Code of conduct
├── _static/             # Static assets (CSS, images)
├── conf.py              # Sphinx configuration
├── Makefile             # Build commands
├── make.bat             # Windows build commands
└── requirements.txt     # Python dependencies
```

## Contributing to Documentation

### Adding New Tutorials

1. Create a new Jupyter notebook in `examples/`
2. Update `examples/index.md` to include the new tutorial
3. Follow the existing notebook structure with markdown and code cells

### Updating References

1. Add new references to `references.bib`
2. Update `references.md` to include the new references
3. Use the `{cite}` directive in markdown files to cite references

### Styling

- CSS customizations go in `_static/custom.css`
- Images and other static assets go in `_static/`
- Logo files should be placed in `_static/` and referenced in `conf.py`

## Automated Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by the GitHub Actions workflow in `.github/workflows/docs.yml`.

## Local Development

For local development, you can use the build script with the `--serve` flag to automatically open the documentation in your browser:

```bash
python build_docs.py --serve
```

This will build the documentation and serve it at `http://localhost:8000`.

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Build failures**: Try cleaning the build directory with `make clean`
3. **Missing images**: Check that all image files are in the `_static/` directory
4. **Notebook execution**: Notebooks are set to not execute by default (`nbsphinx_execute = "never"`)

### Getting Help

If you encounter issues building the documentation:

1. Check that all dependencies are installed correctly
2. Ensure you're using Python 3.12+
3. Try building with verbose output: `make html SPHINXOPTS="-v"`
4. Check the build logs for specific error messages 