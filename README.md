# Yale Astro 330 Labs

Practice projects based on the public materials of **Yale Astro 330** https://astro-330.github.io/intro.html, a course on astronomical data analysis and Python-based research workflows.

This repository records my attempt to learn the basic computational tools used in observational astronomy, including FITS files, photometry, spectroscopy, MCMC, catalog handling, and reproducible scientific computing.

---

## Purpose

My background is not originally in astronomy or physics, so I use these labs as structured training in astronomical data analysis.

The main purpose of this repository is to practice how real astronomical data are stored, reduced, visualized, modeled, and interpreted using Python.

---

## Topics Covered

### 1. Scientific Python Basics

- NumPy arrays
- Plotting with Matplotlib
- Curve fitting
- Basic code organization
- Conda environment setup

### 2. FITS Images and Astronomical Imaging

- Reading FITS files
- Displaying astronomical images
- Working with WCS information
- Basic image scaling and visualization
- Aperture photometry

### 3. Photometry Pipeline Practice

- Background estimation
- Source detection
- Centroiding
- PSF-related measurements
- Basic pipeline structure using Python classes

### 4. Model Fitting and MCMC

- Chi-square fitting
- Likelihood, prior, and posterior
- Simple Metropolis-Hastings sampling
- Using `emcee` for MCMC
- Comparing deterministic fitting with Bayesian sampling

### 5. Stellar Spectroscopy

- Reading observed and synthetic spectra
- Matching continuum levels
- Smoothing spectra to instrumental resolution
- Estimating radial velocity through spectral fitting

### 6. Catalog Analysis

- Working with Pandas DataFrames
- Filtering and joining catalogs
- Basic analysis of astronomical survey data
- Visualization of galaxy or stellar samples

---

## Skills Practiced

- Python-based astronomical data analysis
- FITS file handling
- Photometry and spectroscopy workflows
- MCMC and statistical fitting
- Catalog manipulation
- Scientific visualization
- Reproducible notebook-based analysis

---

## Why This Repository Matters

This repository is not a finished research project. It is a training record.

It helped me move from general Python data analysis toward astronomical workflows, especially the connection between:

**data format → measurement → model fitting → physical interpretation**

These skills are directly relevant to future projects involving stellar spectra, survey data, and observational astrophysics.

---

## Tools

- Python
- NumPy
- SciPy
- Pandas
- Astropy
- Matplotlib
- emcee
- Jupyter Notebook

---

## Status

This repository is a learning repository. I will continue to clean the notebooks, improve documentation, and reuse parts of this workflow in more focused astrophysics projects.
