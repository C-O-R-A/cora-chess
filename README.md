
# Cobot chess

Chess applicaition of the CORA cobot arm. It uses monocular computer vision to localize the chessboard w.r.t. itself. The chess engine uses a cnn to rank all possible legal moves and chooses to play the move ranked the highest. Each 'agent' has been trained and biased towards the moves of a single player respectively. Then, based on the transforms between the chessboard and the robot end effector, a pick and place command is sent to the robot using its ethernet sdk to make its move on the physical chessboard.

---

## Project Structure

```bash
assets/
chess/
    agents/
    data/
    markers/
    models/
    training/
    agents.py
    main.py/
    transforms.py/
    vision.py/
test/
pyproject.toml
requirements.txt
```

---

## Installation

### Clone directly from GitHub

```bash
git clone <url> >> <your directory>

```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dependencies

* `numpy`
* `chess`
* `codi` (networking sdk)
* `torch`
* `pandas`
* `seaborn`

All will be installed automatically when installing via requirements.txt.

---

## Basic Usage

### Camera Calibration

```python 
from vision import camera_calibration

# Use camera 0 and use the checkerboard
camera_calibration(0, use_checkerboard=True)
```


### Marker Placement

![Colossus Logo](assets/board.png)


### Playing a match
