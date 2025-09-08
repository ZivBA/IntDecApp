# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3D Scene Reconstruction Application designed to transform photos and videos into immersive, navigable 3D environments. The project is currently in early development with just a basic Python setup.

## Development Commands

### Environment Setup
```bash
# Using uv (preferred)
uv sync

# Traditional pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Code Quality
```bash
# Linting with ruff
uv run ruff check
uv run ruff check --fix  # Auto-fix issues

# Format code
uv run ruff format
```

### Running the Application
```bash
# Run main application
uv run python main.py
```

### Testing
```bash
# Run tests (once test framework is added)
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=src tests/
```

## Architecture Overview

The project follows the structure outlined in the README with planned modules:

- **Core Processing**: Feature extraction, camera calibration, 3D reconstruction
- **Preprocessors**: Video/image processing, panorama handling  
- **ML Models**: Depth estimation, NeRF, Gaussian Splatting (optional)
- **API Layer**: FastAPI server for web interface
- **Visualization**: Scene export and rendering

### Technology Stack
- **Backend**: Python with OpenCV, COLMAP/OpenMVG, Open3D
- **Web**: FastAPI + Three.js for 3D visualization
- **Optional ML**: NeRF, Gaussian Splatting, MiDaS depth estimation
- **Build System**: uv for dependency management, ruff for linting

## Current State

The project is currently minimal with:
- Basic Python project structure (`pyproject.toml`, `main.py`)
- Development dependencies configured (ruff for linting)
- Comprehensive README with detailed architecture plans
- No actual implementation yet - just a "Hello World" placeholder in `main.py`

## Development Notes

- Uses `uv` for fast Python package management
- Ruff configured for linting and formatting in `pyproject.toml`
- Project requires Python >=3.12
- Extensive roadmap in README covers 4 development phases
- Plans include Docker containerization and multi-platform deployment (web + Android)