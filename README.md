# Singular Count Recognition

Object detection and per-class counting with YOLOv8 (Ultralytics) and OpenCV. Given an image, the script detects objects, counts them by class, draws the boxes with a count label per class, and shows the result. A video helper saves an annotated video with the same per-frame counts. The work is also documented in a Jupyter notebook.

## Features

- YOLOv8 detection with confidence threshold 0.5
- Per-class counts for each image (for example `person = 3`) drawn on the output
- Image mode: annotated image shown with matplotlib and class counts returned
- Video mode: annotated frames written to an MP4 file (`process_video` function)
- Notebook (`Cumulative counting detection.ipynb`) with image tests and a video testing helper

## Project structure

- `test.py` - main script: `process_image()` and `process_video()` functions; runs image detection on `obj.jpg`
- `Singular count recognition/` - project folder
  - `Cumulative counting detection.ipynb` - notebook version with image tests (`obj.jpg`, `OIP.jpeg`, `dogs.jpg`) and a video run
  - `test.py` - copy of the image/video script
  - `obj.jpg`, `dogs.jpg`, `OIP.jpeg` - sample images
  - `output_video.mp4` - sample annotated output
  - `yolo11n.pt`, `yolov8n.pt`, `yolov8m.pt` - YOLO weights
  - `.ipynb_checkpoints/` - Jupyter autosave files (can be ignored)

## Requirements

Packages imported by the scripts:

- ultralytics (`YOLO`)
- opencv-python (`cv2`)
- matplotlib
- standard library: `collections.Counter`

Install:

```bash
pip install ultralytics opencv-python matplotlib
```

## Installation

```bash
pip install ultralytics opencv-python matplotlib
```

## Usage

Run the image example from the project folder (it expects `obj.jpg` and `yolov8m.pt` in the working directory):

```bash
python test.py
```

- Input: image file (`obj.jpg`); the image is resized to 640x480 before detection
- Output: matplotlib window with the annotated image, and the class counts printed to the console

To process a video, call the function from Python (it is not called by default):

```python
from test import process_video
process_video("input.mp4", "output.mp4", "yolov8n.pt", output_resolution=(640, 480))
```

- Input: any video file path
- Output: MP4 file written to `output_path`; the frames are also shown in a window (press `q` to stop early)

## Notes

- The notebook contains a video path from the original author's Windows machine (`C:\Users\...\counting.mp4`). Replace it with your own video.
- Counts are per frame, not cumulative object tracking, despite the notebook name "Cumulative counting". No tracker is used.
- The output video is written at the resolution given to `process_video`, and frames are resized to that size before detection.
- Weight files are included in the folder. The `.ipynb_checkpoints` folder is an artefact of Jupyter and is not needed.
