# Spawn a character into a video with Wan 2.2 Animate

Uses [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) (Apache 2.0).
Give it one **character image** and one **source video**:

| Mode | Result |
|------|--------|
| `replace` | The person in the video is swapped for your character; scene, lighting and camera stay. |
| `animate` | Your character performs the motion from the video. |

## Requirements

- Linux + NVIDIA GPU. An 80 GB card (A100/H100) is the safe choice for the 14B model; multi-GPU also works.
- ~80 GB free disk for weights, Python 3.10+, `git`, `ffmpeg`.
- Works on RunPod / Lambda / Vast / Colab Pro (A100).

## 1. Setup (once)

```bash
cd wan-animate
./setup.sh
```

Clones `Wan-Video/Wan2.2`, installs dependencies, downloads the weights
(Hugging Face, falling back to ModelScope).

## 2. Spawn your character

```bash
# Replace the person in the video with your character
./spawn_character.sh --mode replace --video clip.mp4 --image character.png

# Make your character do the video's motion
./spawn_character.sh --mode animate --video dance.mp4 --image character.png

# Options
#   --out path.mp4      output file (default: outputs/<video>_<image>_<mode>.mp4)
#   --res 1280x720      processing resolution (use 832x480 for faster/lower memory)
#   --multi-gpu 8       FSDP + Ulysses across N GPUs
```

Tips: use a full-body, clearly lit character image whose pose roughly matches the
person in the video; short clips (≤5 s) generate fastest.
