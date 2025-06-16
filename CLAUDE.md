# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
- Main entry point: `python main.py` - processes all XML horse racing data and extracts features
- Uses uv for dependency management: `uv sync` to install dependencies

### Code Quality
- Code formatting: `black .` (Black formatter is configured as a dependency)

## Architecture Overview

This is a horse racing data processing and feature engineering system that:

1. **Data Input**: Processes horse racing XML files from Equibase located in `datasets/` organized by date (e.g., `datasets/2025_06/2025_06_06/`)
2. **Data Extraction**: `main.py` parses XML files using `tags_selector.json` configuration to extract relevant race, track, and horse entry data
3. **Feature Engineering**: `FeatureProcessor` class transforms raw data into ML-ready features including:
   - Distance conversions (furlongs)
   - Field size calculations
   - Odds rankings
   - Days since last race
   - Trainer win percentages

### Key Components

- **main.py**: Main orchestrator that processes XML files and coordinates feature extraction
- **muffin_horsey/feature_processor.py**: Core feature engineering logic using Polars DataFrames
- **tags_selector.json**: XML tag mapping configuration for data extraction
- **sample_training.yaml**: Sample training data structure showing expected feature format

### Data Flow
1. XML files → `process_xml()` → structured dict data
2. Multiple XML files → `merge_xml()` → consolidated Polars DataFrame  
3. Raw DataFrame → `FeatureProcessor.extract_features()` → ML-ready features

### Dependencies
- **Polars**: Primary data processing library (not pandas)
- **Selenium ecosystem**: For web scraping (undetected-chromedriver, selenium-wire)
- **Requests**: HTTP requests for data downloading

## Development Preferences

- **Show, don't modify**: When suggesting code changes, display the proposed code rather than directly editing files
- User prefers to review and apply changes manually