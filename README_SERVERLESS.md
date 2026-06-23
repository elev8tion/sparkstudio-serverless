# SparkStudio Serverless Containers

CI repo for building and pushing SparkStudio's GPU inference containers to Docker Hub.

## Containers

| Container | Model | Docker Hub | Purpose |
|-----------|-------|------------|---------|
| Lip Sync | Wav2Lip | `anomaly13/sparkstudio-lipsync` | Portrait + audio → lip-synced video |
| Sound Effects | MusicGen-Small | `anomaly13/sparkstudio-sfx` | Text → SFX / ambient audio |

## Build

Pushed to `main` triggers GitHub Actions to build and push both images to Docker Hub.

## Deploy

Images are deployed as [Novita Serverless GPU](https://novita.ai/gpus-console/serverless) endpoints that auto-scale to zero.
