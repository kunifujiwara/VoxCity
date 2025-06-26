# VoxCity GitHub Actions Workflows

This directory contains the GitHub Actions workflows for the VoxCity project, providing automated CI/CD, testing, documentation building, and release management.

## Workflows Overview

### 1. **CI Workflow** (`ci.yml`)
**Triggers:** Push to main/master, Pull requests
**Purpose:** Continuous Integration testing and quality assurance

**Features:**
- **Multi-platform testing:** Ubuntu, Windows, macOS
- **Multi-Python version testing:** Python 3.10, 3.11, 3.12
- **Code quality checks:** Linting with Ruff, type checking with MyPy
- **Test coverage:** Generates coverage reports and uploads to Codecov
- **Package building:** Builds the package for distribution
- **Dependency caching:** Caches pip dependencies for faster builds

**Jobs:**
- `test`: Runs tests across multiple platforms and Python versions
- `lint`: Performs code linting and type checking
- `build`: Builds the package (runs after test and lint pass)

### 2. **Documentation Workflow** (`docs.yml`)
**Triggers:** Changes to docs/, src/, README.md, or pyproject.toml
**Purpose:** Build and deploy documentation to GitHub Pages

**Features:**
- **Automatic documentation building:** Builds Sphinx documentation
- **GitHub Pages deployment:** Automatically deploys to GitHub Pages
- **Dependency caching:** Caches documentation dependencies
- **Path-based triggers:** Only runs when documentation-related files change
- **Custom domain support:** Configurable for custom documentation domains

**Jobs:**
- `build-docs`: Builds HTML documentation and deploys to GitHub Pages

### 3. **Release Workflow** (`release.yml`)
**Triggers:** New GitHub release published
**Purpose:** Automated package publishing to PyPI

**Features:**
- **PyPI publishing:** Automatically publishes to PyPI when a release is created
- **GitHub release creation:** Creates GitHub releases with release notes
- **Secure token handling:** Uses encrypted secrets for PyPI authentication
- **Build verification:** Ensures package builds correctly before publishing

**Jobs:**
- `deploy`: Builds and publishes the package to PyPI

## Workflow Dependencies

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Push/PR   │───▶│     CI      │───▶│   Release   │
│             │    │  (ci.yml)   │    │ (release.yml)│
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       └───────────▶│   Docs      │
                    │ (docs.yml)  │
                    └─────────────┘
```

## Configuration Requirements

### Repository Secrets
The following secrets need to be configured in your GitHub repository settings:

1. **`PYPI_API_TOKEN`** (for release workflow)
   - Generate from PyPI account settings
   - Used for publishing packages to PyPI

2. **`GITHUB_TOKEN`** (automatically provided)
   - Used for GitHub Pages deployment
   - Used for creating GitHub releases

### Environment Setup
- **GDAL dependencies:** All workflows install GDAL system dependencies
- **Python versions:** Supports Python 3.10, 3.11, and 3.12
- **Platforms:** Ubuntu, Windows, and macOS support

## Usage

### For Contributors
1. **Fork and clone** the repository
2. **Create a feature branch** for your changes
3. **Make changes** and push to your fork
4. **Create a pull request** - CI will automatically run
5. **Wait for CI to pass** before merging

### For Maintainers
1. **Merge pull requests** after CI passes
2. **Create a release** on GitHub when ready to publish
3. **Documentation** will automatically update on GitHub Pages
4. **Package** will automatically publish to PyPI

### For Documentation Updates
1. **Edit files** in the `docs/` directory
2. **Push changes** to main/master branch
3. **Documentation** will automatically rebuild and deploy

## Customization

### Adding New Workflows
1. Create a new `.yml` file in `.github/workflows/`
2. Follow the existing workflow patterns
3. Test locally using `act` if needed

### Modifying Existing Workflows
1. Update the workflow file
2. Test with a pull request
3. Monitor the Actions tab for any issues

### Environment-Specific Configurations
- **Development:** Use feature branches and pull requests
- **Staging:** Use release candidates or beta releases
- **Production:** Use official GitHub releases

## Troubleshooting

### Common Issues
1. **GDAL installation failures:** Check system dependency installation
2. **Python version conflicts:** Ensure compatibility with supported versions
3. **Documentation build failures:** Check Sphinx configuration and dependencies
4. **PyPI publishing failures:** Verify API token and package configuration

### Debugging
1. **Check Actions tab:** View detailed logs for each workflow run
2. **Local testing:** Use `act` to test workflows locally
3. **Dependency issues:** Check `requirements.txt` and `requirements_dev.txt`
4. **Configuration errors:** Verify workflow syntax and secrets

## Best Practices

1. **Keep workflows focused:** Each workflow should have a single responsibility
2. **Use caching:** Cache dependencies to speed up builds
3. **Fail fast:** Stop builds early if critical checks fail
4. **Security first:** Use encrypted secrets for sensitive data
5. **Documentation:** Keep this README updated with workflow changes

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages) 