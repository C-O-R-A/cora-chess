
# Cobot chess

Chess applicaition of the CORA cobot arm. It uses monocular computer vision to localize the chessboard w.r.t. itself. The chess engine uses a cnn to rank all possible legal moves and chooses to play the move ranked the highest. Each 'agent' has been trained and biased towards the moves of a single player respectively. Then, based on the transforms between the chessboard and the robot end effector, a pick and place command is sent to the robot using its ethernet sdk to make its move on the physical chessboard.

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
## Vision and Perception
This section covers the vision and perception pipelines used for detecting the board and square positions as well as the moves made by the opponent.

### Camera Calibration

```python 
from vision import camera_calibration

# Use camera 0 and use the checkerboard
camera_calibration(0, use_checkerboard=True)
```


### Marker Placement

![Board](assets/board.png)




### Transforms
![Triads](assets/Transforms.png)
For this application we assign three triads to the robot ($W, C, G$) and five 
to the the markers and boad respectively ($M_{1,2,3,4}, B_O$). From these definitions the following transforms apply:

$
^WT_{B_O} = \begin{bmatrix} {^WR_{B_O}} & {^W\vec{r}_{B_O}} \\\ 0 & 1 \end{bmatrix} =  {^WT_{C}}  {^CT_{B_O}} \\[1em]
{^CT_{B_O}} = {^CT_{M_i}} \space {^{M_i}T_{B_O}} \qquad 1 \le i \le 4 \\[1em]
$

And with $^{M_1}\vec{r}_{B_O} = \begin{bmatrix} {t} \\\ {t} \\\ {0} \end{bmatrix}$,

$
{^{M_1}T_{B_O}} = \begin{bmatrix} {0} & 0 & 0 & t \\\ 0 & 0 & 0 & t \\\ 0 & 0 & 0 & 0 \\\ 0 & 0 & 0 & 1\end{bmatrix}
$

---
### Basic Usage
---
### Playing a match
