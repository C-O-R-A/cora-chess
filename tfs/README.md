
## Transforms  
<img src="assets/Transforms.png">



## Coordinate Frames

#### Robot Frames
- <img src="assets/image-4.png" width="80" style="vertical-align: middle;"> World (robot base) frame  
- <img src="assets/image-5.png" width="80" style="vertical-align: middle;"> Camera frame  
- <img src="assets/image-6.png" width="80" style="vertical-align: middle;"> Gripper (end-effector) frame  

#### Environment Frames
- <img src="assets/image-7.png" width="220" style="vertical-align: middle;"> Fiducial marker frames  
- <img src="assets/image-8.png" width="80" style="vertical-align: middle;"> Chessboard origin frame  

---

## Transform Chains

#### Camera to Board Origin

The pose of the chessboard in the world frame is obtained via composition in $\mathrm{SE}(3)$:

<img src="assets/image-3.png" width="230">

The camera-to-board transform is computed from the detected fiducial markers:

<img src="assets/image-2.png" width="400">

---

#### Marker Offset

The chessboard origin is defined relative to marker $M_1$ by a fixed translation:

<img src="assets/image-1.png" width="150">

yielding the homogeneous transform:

<img src="assets/image.png" width="280">
