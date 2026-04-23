# 👤 GhostMask

**Real-time face blur & identity protection tool.**  
Automatically detects and pixelates faces in live webcam feeds, video files, and still images — with zero data stored, zero network calls, and the lowest possible latency.

---

## Why GhostMask?

Publishing footage of protests, public events, or street photography raises real privacy concerns. GhostMask solves this in seconds — no account, no cloud, no trace.

---

## Features

- ⚡ **Real-time** — processes live webcam at 30–60 FPS on a standard laptop
- 🎭 **Partial face detection** — catches side profiles and half-occluded faces
- 🔵 **Oval feathered mask** — smooth elliptical blur, not harsh rectangles
- 🎨 **Three blur modes** — pixel mosaic, Gaussian, box blur
- 🔒 **100% local** — no biometric data stored or transmitted, ever
- 🖥️ **Three input modes** — webcam, video file, single image

---

## Project Structure

```
ghostmask/
│
├── main.py            ← CLI entry point
├── face_detector.py   ← MediaPipe BlazeFace wrapper
├── blur_engine.py     ← Blur algorithms + oval masking
├── processor.py       ← Pipeline orchestration (webcam / video / image)
├── requirements.txt   ← Python dependencies
└── GUIDE.md           ← Full A–Z setup & usage guide
```

---

## How It Works

```
Input (webcam / video / image)
         │
         ▼
   Face Detection          ← MediaPipe BlazeFace (~1–3 ms/frame)
   (bounding boxes)
         │
         ▼
   Blur Engine             ← Pixel mosaic applied to face ROI only
   + Oval Mask             ← Elliptical feathered blend
         │
         ▼
   Output frame / file
```

**Detection:** Google's BlazeFace neural network — purpose-built for real-time use, runs entirely on CPU, handles partial and side-view faces.

**Blurring:** Only the face region-of-interest (ROI) is processed, not the full frame, keeping latency minimal.

---

## Quick Start

```bash
# 1. Clone and enter the project
cd ghostmask

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py webcam           # live webcam
python main.py video input.mp4  # process a video file
python main.py image photo.jpg  # process a single image
```

---

## Usage Examples

```bash
# Webcam — catch partial/profile faces (lower confidence threshold)
python main.py webcam --confidence 0.3 --block-size 20

# Webcam — Gaussian blur, oval mask
python main.py webcam --mode gaussian --strength 75

# Video — process and save
python main.py video footage.mp4 --output safe_footage.mp4

# Image — rectangular blur zone, no preview window
python main.py image portrait.jpg --shape rect --no-show
```

### Webcam Hotkeys

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot |
| `O` | Toggle overlay |
| `B` | Toggle debug boxes |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera I/O, video, image drawing |
| `mediapipe` | BlazeFace face detection |
| `numpy` | Fast array math |
| `Pillow` | Image format support |
| `tqdm` | Video processing progress bar |

---

## Configuration Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model 0\|1` | `0` | `0` = fast short-range, `1` = full-range (crowds) |
| `--confidence` | `0.4` | Lower = catch more partial faces |
| `--padding` | `0.20` | Extra space added around each face box |
| `--mode` | `pixel` | `pixel`, `gaussian`, or `box` |
| `--block-size` | `15` | Mosaic tile size (pixel mode) |
| `--strength` | `55` | Kernel size (gaussian/box modes) |
| `--shape` | `oval` | `oval` (feathered ellipse) or `rect` |

See `GUIDE.md` for the complete A–Z walkthrough.

---

## License

For personal and journalistic privacy protection use. Do not use to process footage you don't have rights to.
