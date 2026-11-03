# AI Interview Coach

AI Interview Coach is a mobile app built with Expo and React Native for college project demos. It lets a user upload or paste a resume, generates interview questions from that resume, simulates a chat and voice interview coach, and shows feedback with improvement suggestions.

## Features

- Resume upload or pasted resume text
- Resume keyword extraction and dynamic question generation
- Voice prompt playback for interview practice
- Chat-style interview coaching
- Answer scoring and feedback suggestions
- Resume-based follow-up questions

## Tech Stack

- Expo
- React Native
- TypeScript
- React Navigation
- expo-speech
- expo-document-picker
- expo-file-system
- expo-av

## Setup

1. Install dependencies:

```bash
npm install
```

2. Start the app:

```bash
npm run start
```

3. Open it in Expo Go or run on an emulator.

## Notes

- PDF parsing is intentionally lightweight in this demo.
- For best results, paste resume text or use a text-based resume file.
- The interview scoring is local and rule-based so the app works without an API key.

## Steel Defect Diagnosis - Phase 1 Foundation

This workspace now also contains the foundation for the final-year project:

**AI-Driven Multi-Stage Steel Defect Diagnosis and Evidence-Based Corrective Action Recommendation System**

Current scope is Phase 1 foundation only:

- environment verification
- NEU-DET dataset inspection
- reproducible project folders
- no model training yet
- no Phase 2 corrective-action recommendation system

### Expected Dataset Location

Place the NEU-DET dataset locally at:

```text
data/raw/NEU-DET/
```

Expected simple class-folder structure:

```text
data/raw/NEU-DET/
  crazing/
  inclusion/
  patches/
  pitted_surface/
  rolled-in_scale/
  scratches/
```

If your downloaded NEU-DET copy uses a different folder layout, run the inspection script with `--dataset-root` pointing to that folder first. Do not modify the original dataset.

### Python Setup

After installing Python 3.10 or 3.11, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:

```powershell
python scripts/verify_environment.py
```

Inspect the dataset:

```powershell
python scripts/inspect_dataset.py
```

The dataset inspection script writes inventory and summary files to `results/metrics/dataset_inspection/` and report-ready figures to `results/figures/dataset_inspection/`.

## Project Structure

- `App.tsx` wires navigation and theme.
- `src/state/InterviewContext.tsx` holds shared interview state.
- `src/services/interviewEngine.ts` generates questions and feedback.
- `src/screens/` contains the main app screens.
