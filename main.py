"""
main.py
─────────────────────────────────────────────────────────────────────────────
CLI entry point for the Face Blur / Identity Protection Tool.

Quick start
───────────
  python main.py webcam                          # live webcam, defaults
  python main.py webcam --camera 1 --mode pixel  # second camera
  python main.py video  input.mp4                # process video file
  python main.py image  photo.jpg                # process single image

Full help
─────────
  python main.py --help
  python main.py webcam --help
"""

import argparse
import sys
from processor import run_webcam, run_video, run_image


# ─────────────────────────────────────────────── shared args ─── #

def _add_common(p: argparse.ArgumentParser):
    p.add_argument(
        "--model", type=int, default=0, choices=[0, 1],
        help="MediaPipe model: 0=short-range fast (default), 1=full-range",
    )
    p.add_argument(
        "--confidence", type=float, default=0.4, metavar="0-1",
        help="Min detection confidence. Lower → catch more partial faces. (default 0.4)",
    )
    p.add_argument(
        "--padding", type=float, default=0.20, metavar="0-1",
        help="Fractional padding added around each face box. (default 0.20)",
    )
    p.add_argument(
        "--mode", choices=["pixel", "gaussian", "box"], default="pixel",
        help="Blur algorithm: pixel (mosaic), gaussian, box. (default pixel)",
    )
    p.add_argument(
        "--block-size", type=int, default=15,
        help="Mosaic tile size in pixels – pixel mode only. (default 15)",
    )
    p.add_argument(
        "--strength", type=int, default=55,
        help="Kernel size for gaussian/box modes (must be odd). (default 55)",
    )
    p.add_argument(
        "--shape", choices=["oval", "rect"], default="oval",
        help="Blur region shape: oval (elliptical mask) or rect. (default oval)",
    )


# ─────────────────────────────────────────────── sub-commands ─── #

def _webcam_cmd(args):
    run_webcam(
        camera_index=args.camera,
        model_selection=args.model,
        min_confidence=args.confidence,
        padding=args.padding,
        blur_mode=args.mode,
        block_size=args.block_size,
        strength=args.strength,
        shape=args.shape,
        show_overlay=not args.no_overlay,
        show_boxes=args.boxes,
        width=args.width,
        height=args.height,
    )


def _video_cmd(args):
    run_video(
        input_path=args.input,
        output_path=args.output,
        model_selection=args.model,
        min_confidence=args.confidence,
        padding=args.padding,
        blur_mode=args.mode,
        block_size=args.block_size,
        strength=args.strength,
        shape=args.shape,
        preview=args.preview,
    )


def _image_cmd(args):
    run_image(
        input_path=args.input,
        output_path=args.output,
        model_selection=args.model,
        min_confidence=args.confidence,
        padding=args.padding,
        blur_mode=args.mode,
        block_size=args.block_size,
        strength=args.strength,
        shape=args.shape,
        show=not args.no_show,
    )


# ─────────────────────────────────────────────────── main ─── #

def main():
    parser = argparse.ArgumentParser(
        prog="faceblur",
        description="🎭  Real-time / offline face blur & identity protection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  python main.py webcam
  python main.py webcam --confidence 0.3 --mode pixel --block-size 20
  python main.py video  recording.mp4 --output safe.mp4
  python main.py image  photo.jpg --shape rect --mode gaussian
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── webcam ──────────────────────────────────────────────────────────
    p_cam = sub.add_parser("webcam", help="Live webcam processing")
    _add_common(p_cam)
    p_cam.add_argument("--camera", type=int, default=0,
                       help="Camera device index (default 0)")
    p_cam.add_argument("--width",  type=int, default=1280)
    p_cam.add_argument("--height", type=int, default=720)
    p_cam.add_argument("--no-overlay", action="store_true",
                       help="Hide FPS/face-count overlay")
    p_cam.add_argument("--boxes", action="store_true",
                       help="Draw raw detection boxes (debug)")
    p_cam.set_defaults(func=_webcam_cmd)

    # ── video ───────────────────────────────────────────────────────────
    p_vid = sub.add_parser("video", help="Process a video file")
    _add_common(p_vid)
    p_vid.add_argument("input", help="Path to input video")
    p_vid.add_argument("--output", "-o", default=None,
                       help="Output path (default: <input>_blurred.mp4)")
    p_vid.add_argument("--preview", action="store_true",
                       help="Show live preview window while processing")
    p_vid.set_defaults(func=_video_cmd)

    # ── image ───────────────────────────────────────────────────────────
    p_img = sub.add_parser("image", help="Process a single image")
    _add_common(p_img)
    p_img.add_argument("input", help="Path to input image (jpg/png/…)")
    p_img.add_argument("--output", "-o", default=None,
                       help="Output path (default: <input>_blurred.<ext>)")
    p_img.add_argument("--no-show", action="store_true",
                       help="Don't open a preview window after processing")
    p_img.set_defaults(func=_image_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
