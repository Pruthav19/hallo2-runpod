# Hallo4 H100 RunPod Test Branch

This branch adds a separate Hallo4 test setup on top of the existing `hallo2-runpod` repository.

Branch: `test/hallo4-h100-runpod`

## What is included

- `Dockerfile.hallo4` — separate RunPod image for Hallo4 on CUDA 12.1 / PyTorch 2.5.1.
- `download_hallo4_models.py` — downloads `fudan-generative-ai/hallo4` into `/runpod-volume/pretrained_models_hallo4`.
- `start_hallo4.sh` — starts model download and then the RunPod handler.
- `handler_hallo4.py` — minimal smoke-test handler to verify build, GPU, and model files.

## Why this is a smoke test first

Hallo4 upstream currently has two important RunPod portability issues:

1. `requirements.txt` includes local `file:///cpfs...` wheel paths for `flash_attn` and `wan`. Those paths only exist in the authors' environment and cannot work directly on RunPod.
2. The README says it was tested on H100, but does not provide a RunPod-ready serverless handler.

So this branch first validates that the container builds, the H100 is visible, and all model files are cached correctly. After that passes, the next step is to wire the full inference command.

## Build image

From the repo root:

```bash
docker build -f Dockerfile.hallo4 -t hallo4-h100-runpod:test .
```

For reproducible builds, pin the upstream commit:

```bash
docker build \
  --build-arg HALLO4_REF=<upstream-hallo4-commit-sha> \
  -f Dockerfile.hallo4 \
  -t hallo4-h100-runpod:test .
```

## RunPod setup

Use an H100 pod/serverless endpoint with a network volume mounted at:

```text
/runpod-volume
```

On first boot, models are downloaded to:

```text
/runpod-volume/pretrained_models_hallo4
```

Expected model layout follows the upstream Hallo4 README:

```text
pretrained_models_hallo4/
|-- hallo4/model_weight.pth
|-- Wan2.1_Encoders/Wan2.1_VAE.pth
|-- Wan2.1_Encoders/models_t5_umt5-xxl-enc-bf16.pth
|-- audio_separator/Kim_Vocal_2.onnx
|-- wav2vec/wav2vec2-base-960h/config.json
```

## Smoke-test input

Send any JSON input. The handler currently ignores the payload and returns environment checks.

Expected success response:

```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu_name": "NVIDIA H100 ...",
  "checks": {
    "hallo4_dir_exists": true,
    "inference_module_exists": true,
    "model_weight_exists": true,
    "wan_vae_exists": true,
    "audio_separator_exists": true,
    "wav2vec_exists": true
  }
}
```

## Full inference command to wire next

Upstream inference uses:

```bash
python -m vace.vace_wan_inference \
  --prompt "a person is talking" \
  --src_video assets/01.mp4 \
  --src_ref_images assets/01.png \
  --src_audio assets/01.wav \
  --save_dir outputs \
  --ckpt_dir /runpod-volume/pretrained_models_hallo4/Wan2.1_Encoders \
  --model_path /runpod-volume/pretrained_models_hallo4/hallo4/model_weight.pth \
  --audio_separator_model_path /runpod-volume/pretrained_models_hallo4/audio_separator/Kim_Vocal_2.onnx \
  --wav2vec_model_path /runpod-volume/pretrained_models_hallo4/wav2vec/wav2vec2-base-960h
```

Important: Hallo4 requires WAV audio and English audio according to upstream README. For best results, provide a real source/driving video instead of a static placeholder.

## Known risks

- `flash-attn` build may fail depending on the RunPod build environment. The Dockerfile currently continues if it fails so smoke testing can proceed.
- The upstream `wan` dependency is referenced as a local wheel in the authors' requirements. If the cloned Hallo4 repo does not include importable WAN/VACE modules, full inference may require installing the public WAN2.1 package or patching imports.
- This branch does not replace the existing Hallo2 worker.
