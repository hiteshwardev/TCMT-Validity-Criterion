# TCMT validity criterion

Code and data for the paper *Accuracy limits of single-mode temporal coupled-mode theory under
coherent two-port excitation* (H. K. Singh).

The study measures how accurately single-mode temporal coupled-mode theory (TCMT) reproduces the
exact scattering matrix of a lossless two-port when both ports are driven coherently. Exact
reference spectra come from a transfer-matrix solver (Fabry–Perot cavities with Bragg mirrors)
and a Fourier modal method (binary dielectric gratings). The notebook derives and verifies a
closed form for the worst case over the excitation phase, fits a criterion for the worst-case
error as a function of the mode separation in linewidths, explains the error through the tails of
the neighbouring resonances, and tests the criterion on guided-mode-resonance gratings.

## Contents

```
notebook.ipynb     the complete study: theory, validation, computations, figures, final numbers
src/tcmtcrit/      modules used throughout the notebook
  config.py          fixed parameters, tolerances and seeded random generators
  tmm.py             transfer-matrix method for planar multilayers
  cavity.py          Bragg-mirror Fabry–Perot cavities, mode catalogue, measurement window
  fmm.py             Fourier modal method for one-dimensional gratings (TE polarisation)
  tcmt.py            single-mode and multimode TCMT models and their least-squares fits
  observables.py     coherent-drive error functional, quadratic forms and the phase law
  criterion.py       criterion fits, model selection, cross-validation and bootstrap
  figstyle.py        figure style and automated layout checks
figures/           all manuscript figures (PNG at 600 dpi and PDF)
results/           all numerical results as JSON; summary.json collects the final numbers
```

## Running

Create the environment with conda or pip:

```
conda env create -f environment.yml
conda activate tcmt-validity
```

or

```
python -m pip install -r requirements.txt
```

Then execute the notebook from the repository root, either interactively or with

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebook.ipynb
```

A full run regenerates every file in `results/` and `figures/` and takes about 70 minutes on
one CPU core. All random draws are derived from the master seed in `src/tcmtcrit/config.py`, so
repeated runs give identical results. The validation section uses the
[grcwa](https://github.com/weiliangjinca/grcwa) package only as an independent cross-check of the
Fourier modal method.

## Results

| File | Content |
|:---|:---|
| `validation.json` | solver and identity benchmarks with their tolerances |
| `cavity_structures.json` | the realised cavity structures, their mode combs and error measures |
| `phase_law.json` | direct and closed-form error against excitation phase |
| `criterion.json` | model selection, leave-one-out validation and bootstrap intervals |
| `tail_strength_cavity.json` | tail strength and its relation to the worst-case error |
| `window_sensitivity.json` | the criterion for alternative measurement windows |
| `multimode.json` | errors of multimode and dispersive-background models |
| `quality_factor.json` | quality-factor sweeps at fixed separation |
| `grating.json`, `grating_summary.json` | grating resonances and the transfer test |
| `summary.json` | all numbers quoted in the paper |

## License

MIT, see `LICENSE`.
