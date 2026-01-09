# Cobot Chess
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

Chess application for the **CORA cobot arm**.  
The system uses **monocular computer vision** to estimate rigid-body transforms in
${\mathrm{SE}(3)}$ between the robot, camera, fiducial markers, and a physical chessboard.
A **CNN-based chess engine** selects moves, which are executed via pick-and-place
motions using the robot’s Ethernet SDK.

Each agent is trained and biased toward the move distribution of a single human
player.


---


## Project Structure

```bash
assets/
chess/
tfs/
vision/
main.py
tests/
pyproject.toml
requirements.txt
```

---

## Installation

### Clone the Repository
```bash
git clone <https://github.com/C-O-R-A/cora-chess.git> <your-directory>
cd <your-directory>
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

- `numpy`
- `chess`
- `codi` (networking SDK)
- `torch`
- `pandas`
- `seaborn`

>All dependencies are installed automatically when using `requirements.txt`.

---

## Vision and Perception

This section describes the perception pipeline for estimating camera pose,
board pose, and opponent moves using rigid-body transforms in
${\mathrm{SE}(3) }$.

---

## Camera Calibration

```python
from vision import camera_calibration

# Use camera 0 with a checkerboard pattern
camera_calibration(0, use_checkerboard=True)
```

---

## Marker Placement

<img src="assets/Transforms.png">

![Board](assets/board.png)

---

## Basic Usage

```bash
python3 main.py
```

The system will:

1. Detect fiducial markers and the chessboard
2. Estimate all required ${\mathrm{SE}(3) }$ transforms
3. Observe the opponent’s move
4. Select a move using the CNN-based engine
5. Execute the move using the robot arm

---

## Playing a Match

Ensure that:

- The camera is calibrated
- All fiducial markers are visible
- The chessboard lies within the robot’s reachable workspace

Once running, the robot alternates turns with the human opponent and physically
moves the chess pieces on the board.

---



