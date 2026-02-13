# 2026-02 Image Manipulations

A learning Computer Vision project: a set of core image manipulation operations implemented with `NumPy`, accelerated with `Numba`, and executed through a configurable pipeline (`pipeline.yaml`).

This project helps you:
- practice implementing CV operations from scratch (without high-level ready-made filters);
- understand how sequential image processing pipelines are built;
- experiment quickly by changing only the config.

## Features

Supported operations:
- `CROP` - crop a region `(y_min, x_min, y_max, x_max)`;
- `ROTATE` - rotate with bilinear interpolation;
- `BLUR` - blur (Gaussian-like ring weights + `Numba`);
- `MIRROR` - mirror by axis (`axis: 0` or `1`);
- `COMPRESSION` - downsample to `size x size` using area downsampling;
- `PERMUTATION` - shuffle `size x size` blocks;
- `INVERT` - invert intensities (`255 - x`).

## Project Structure

- `main.py` - entry point; reads an image, runs pipeline, saves output;
- `pipeline.yaml` - operation order and parameters;
- `src/manipulations.py` - implementations of all manipulations;
- `src/PipelineRunner.py` - config loading/validation and sequential execution;
- `src/utils.py` - shared constants, including default image list;
- `data/raw` - input images;
- `data/output` - output images;
- `logs/app.log` - execution logs.

## Requirements

- Python `3.10+`
- dependencies from `pyproject.toml` (`numpy`, `numba`, `opencv-python`, `pydantic`, `pyyaml`, `matplotlib`, etc.)

## Installation

### Option 1 (recommended): `uv`

```bash
uv sync
```

### Option 2: `venv` + `pip`

If you use `pip`, install dependencies manually from `pyproject.toml` or via an exported requirements file.

## How to Use

1. Put input images into `data/raw`.
2. In `main.py`, select the image via `choose_pic`, for example:
   - `Picture.BOATS`
   - `Picture.DOOR`
   - `Picture.ISLAND`
   - `Picture.MOUNTAINS`
   - `Picture.WHITE`
   - `Picture.GUY`
3. Configure your pipeline in `pipeline.yaml`.
4. Run:

```bash
python main.py
```

After running:
- processed image is saved to `data/output`;
- detailed logs are written to `logs/app.log`.

## `pipeline.yaml` Format

Basic format:

```yaml
pipeline:
  - name: "PERMUTATION"
    params: {size: 256}
  - name: "MIRROR"
    params: {axis: 0}
```

Where:
- `name` - operation name (one of: `CROP`, `ROTATE`, `BLUR`, `MIRROR`, `COMPRESSION`, `PERMUTATION`, `INVERT`);
- `params` - operation parameters.

Parameter examples:
- `CROP`: `{bbox: [50, 100, 500, 700]}`
- `ROTATE`: `{angle: 45}`
- `MIRROR`: `{axis: 1}`
- `COMPRESSION`: `{size: 256}`
- `PERMUTATION`: `{size: 128}`
- `BLUR`, `INVERT`: `{}` (no required parameters)

## Minimal Example Scenario

Goal: shuffle image blocks and mirror the result.
1. Set `PERMUTATION` and `MIRROR` in `pipeline.yaml`;
2. choose an input image in `main.py` (for example, `Picture.ISLAND`);
3. run `python main.py`;
4. check result in `data/output`.

## Common Errors

- `ValueError: Manipulation ... is not supported!`
  - check `name` in `pipeline.yaml` (must match enum operation names).
- Invalid params (`bbox`, `angle`, `size`, `axis`)
  - check value ranges in `src/manipulations.py`.
- File not found
  - ensure the image exists in `data/raw` and filename matches.
