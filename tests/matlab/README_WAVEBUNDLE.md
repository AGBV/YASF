# Wavebundle (Gaussian Beam) Validation

This directory contains MATLAB scripts to generate reference test data using CELES for validating YASFPY's wavebundle implementation.

## Overview

The wavebundle validation compares YASFPY's Gaussian beam initial field coefficients against CELES (MATLAB reference implementation).

**Implementation**: `yasfpy/simulation.py` - `__compute_initial_field_coefficients_wavebundle_normal_incidence()`

**Reference**: [CELES initial_field_coefficients_wavebundle_normal_incidence.m](https://github.com/disordered-photonics/celes/blob/master/src/initial/initial_field_coefficients_wavebundle_normal_incidence.m)

## Prerequisites

### MATLAB Environment
- MATLAB (tested with R2020a or later)
- CELES installed and in MATLAB path
  - Download: https://github.com/disordered-photonics/celes
  - Installation: Follow CELES documentation

### Python Environment
- YASFPY installed with test dependencies
- scipy, numpy, pytest

## Generating Test Data

### Step 1: Run MATLAB Script

```matlab
cd tests/matlab
generate_wavebundle_test_data
```

This script will:
1. Set up a 4-particle test configuration
2. Configure Gaussian beam with beam_width = 400 nm
3. Compute initial field coefficients using CELES
4. Save results to `tests/data/wavebundle_test_data.mat`

**Expected output:**
```
=======================================================
CELES WAVEBUNDLE TEST DATA GENERATION
=======================================================

Particles:
  Number: 4
  Radius: 50.0 nm
  Refractive index: 1.50 + 0.01i

Gaussian beam:
  Beam width: 400.0 nm
  Focal point: [0.0, 0.0, 0.0]
  Polarization: TE
  Polar angle: 0.0 deg

Numerics:
  lmax: 3
  nmax: 30
  Polar angles: 40 points
  Azimuthal angles: 15 points

Computing initial field coefficients...
  ✓ Computation successful
  Initial field coefficients shape: [4, 30]
  Max magnitude: ~5.2e+00

Saving test data...
  ✓ Saved to: ../data/wavebundle_test_data.mat

=======================================================
TEST DATA GENERATION COMPLETE
=======================================================
```

### Step 2: Verify Data File

```bash
ls -lh tests/data/wavebundle_test_data.mat
```

The file should be ~10-50 KB depending on precision.

## Running Validation Test

### Python Test

```bash
# Run validation test
pytest tests/test_wavebundle_validation.py -v -s

# Or with more verbose output
pytest tests/test_wavebundle_validation.py::test_wavebundle_vs_matlab -v -s
```

**Expected output:**
```
=======================================================
WAVEBUNDLE VALIDATION vs CELES/MATLAB
=======================================================

1. Loading MATLAB test data...
   Particles: 4
   lmax: 3
   Wavelength: 550.0 nm
   Beam width: 400.0 nm
   Polarization: TE
   Angular grid: 15 × 40
   MATLAB coefficients shape: (4, 30)

2. Setting up YASFPY simulation...
   ✓ Particles created: 4
   ✓ Gaussian beam configured
   ✓ Numerics with matching angular grid

3. Computing initial field coefficients with YASFPY...
   ✓ YASFPY coefficients shape: (4, 30)
   ✓ Max magnitude: ~5.2e+00

4. Comparing YASFPY vs MATLAB...
   Absolute differences:
   Max: ~1e-3
   Mean: ~1e-4

   Relative differences (for significant coefficients):
   Max: ~1e-3  (0.1%)
   Mean: ~1e-4 (0.01%)

5. Validation...
   ✓ VALIDATION PASSED!
   All coefficients within tolerance (rtol=0.01, atol=0.0001)

=======================================================
VALIDATION COMPLETE
=======================================================

PASSED
```

## Test Configuration

### MATLAB Parameters
- **Particles**: 4 particles in different positions
  - Position 1: [0, 0, 0] (at focal point)
  - Position 2: [100, 0, 0] (off-axis x)
  - Position 3: [0, 100, 0] (off-axis y)
  - Position 4: [0, 0, 200] (along beam axis)
- **Radius**: 50 nm (all particles)
- **Refractive index**: 1.5 + 0.01i
- **Wavelength**: 550 nm
- **Medium**: Vacuum (n = 1.0)
- **Gaussian beam**:
  - Beam width: 400 nm
  - Focal point: [0, 0, 0]
  - Polarization: TE
  - Normal incidence (polar angle = 0°)
- **Angular grid**: 15 azimuthal × 40 polar points

### Validation Tolerances
- **Relative tolerance**: 1% (rtol = 0.01)
- **Absolute tolerance**: 1e-4 (atol = 0.0001)

These tolerances account for:
- Numerical integration differences
- Floating-point precision
- MATLAB vs Python implementation details

## Troubleshooting

### MATLAB Script Fails

**Problem**: `Undefined function 'initial_field_coefficients_wavebundle_normal_incidence'`

**Solution**:
- Ensure CELES is in MATLAB path: `addpath(genpath('/path/to/celes'))`
- Check CELES installation

**Problem**: Out of memory error

**Solution**:
- Reduce angular grid resolution (fewer polar/azimuthal points)
- Use single precision (`single` instead of `double`)

### Python Test Fails

**Problem**: `FileNotFoundError: wavebundle_test_data.mat not found`

**Solution**:
- Run MATLAB script first to generate data
- Verify file location: `tests/data/wavebundle_test_data.mat`

**Problem**: Validation fails with large relative differences (>1%)

**Potential causes**:
1. **Angular grid mismatch**: MATLAB and Python must use identical grids
2. **Integration method**: Check integration implementation
3. **Sign conventions**: Verify phase conventions match CELES
4. **Indexing**: Ensure MATLAB (1-based) vs Python (0-based) handled correctly

**Debugging**:
```python
# Print detailed comparison
pytest tests/test_wavebundle_validation.py::test_wavebundle_vs_matlab -v -s
```

## Customizing Test Data

To test different configurations, edit `generate_wavebundle_test_data.m`:

### Different Beam Width
```matlab
initialField.beamWidth = 200;  % Tighter focus
% or
initialField.beamWidth = 1000;  % Wider beam
```

### Different Polarization
```matlab
initialField.polarization = 'TM';  % Test TM polarization
```

### More Particles
```matlab
particles.position = [
    0, 0, 0;
    100, 0, 0;
    0, 100, 0;
    0, 0, 200;
    150, 150, 0;  % Add more rows
];
% Update radii and refractive indices accordingly
```

### Higher lmax
```matlab
lmax = 5;  % Higher multipole order (slower)
```

After modifications:
1. Re-run MATLAB script to regenerate data
2. Re-run Python validation test

## File Structure

```
tests/
├── matlab/
│   ├── README_WAVEBUNDLE.md           # This file
│   └── generate_wavebundle_test_data.m  # MATLAB data generator
├── data/
│   └── wavebundle_test_data.mat       # Generated test data
└── test_wavebundle_validation.py      # Python validation test
```

## References

- [CELES GitHub](https://github.com/disordered-photonics/celes)
- [CELES Paper](https://arxiv.org/abs/1706.02145)
- YASFPY Implementation: `yasfpy/simulation.py:358-496`
