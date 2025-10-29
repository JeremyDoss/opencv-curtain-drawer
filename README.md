# opencv-curtain-drawer

A real-time interactive curtain effect that responds to human presence using computer vision. For a projection mapping project with James Maker. Dedicated to the late John Thomas Thummel. RIP Homie!

## Features
- Real-time person detection using YOLOv3
- Physics-based curtain animation
- Configurable curtain parameters (gravity, damping, etc.)
- Pure white curtain effect on black background

## Requirements
- Python 3.8 or higher
- Webcam
- YOLOv3 weights and configuration files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JeremyDoss/opencv-curtain-drawer.git
cd opencv-curtain-drawer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv3 model files:
- Download YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights)
- Download YOLOv3-tiny weights from [here](https://pjreddie.com/media/files/yolov3-tiny.weights) (optional, for faster performance)
- Place the downloaded files in the project root directory

## Running the Application

1. Make sure your webcam is connected
2. Run the main script:
```bash
python main.py
```

3. Press 'q' to quit the application

## Configuration

You can adjust various parameters in `main.py`:
- Curtain physics (gravity, damping) in the `CurtainPoint` class
- Detection confidence threshold in the `detect_people` function
- Curtain density by modifying the `spacing` variable in the `main` function
- Camera resolution by uncommenting and adjusting the output width/height settings

## Note

Initially generated with AI. The code may be optimized for performance and line generation improvements in future updates.