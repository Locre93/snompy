# sSNOM Data Analysis & Visualization

A Python package for loading, analyzing, and visualizing scattering-type Scanning Near-Field Optical Microscopy (sSNOM) data obtained with Neaspec near-field optical microscopes. This library allows the user to process GSF scan files, extract channels, perform FFTs, and generate publication-quality plots of amplitude, phase, real, and imaginary components.

**Package name:** `snompy`  
**Version:** 1.0.0  
**Authors:** Lorenzo Orsini, Elisa Mendels, Matteo Ceccanti, Bianca Turini  

---

## Features

- Load `.gsf` files from Neaspec sSNOM measurements
- Handle multiple measurement types:
  - Spatial scans
  - Voltage sweeps
  - Frequency sweeps
- Analyze complex sSNOM signals:
  - Absolute value, phase, real, and imaginary components
  - Normalization, filtering, and sectioning
- Plot 2D maps and cross-sections
- Perform FFT for spatial frequency analysis
- Generate publication-ready plots

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd snompy

## License

This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](LICENSE) file for details.