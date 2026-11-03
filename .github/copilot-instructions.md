# Project Instructions

## Project overview

This repository contains two related demos:

- An Expo/React Native interview coaching app at the repository root and in `src/`.
- A Python steel-defect diagnosis research foundation using the NEU-DET dataset.

## Development conventions

- Keep TypeScript changes compatible with Expo SDK 51 and strict TypeScript checking.
- Keep Python scripts reproducible and use configuration files under `configs/`.
- Do not commit datasets, virtual environments, model checkpoints, framework caches, or generated experiment artifacts.
- Place reusable Python code under `src/`; keep executable workflows under `scripts/`.
- Use the existing formatting, naming, and folder conventions.

## Validation

- TypeScript: `npm run typecheck`
- Python environment: `python scripts/verify_environment.py`
- Dataset-dependent scripts require the local NEU-DET dataset under `data/raw/NEU-DET/`.

## Documentation

Update `README.md` when setup commands, supported workflows, or project scope changes.
