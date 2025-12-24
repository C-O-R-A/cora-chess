## Coordinate Frames

![Triads](assets/Transforms.png)

We define the following coordinate frames:

### Robot Frames
- ${\mathcal{F}_W }$: World (robot base) frame  
- ${\mathcal{F}_C}$ : Camera frame  
- ${\mathcal{F}_G}$ : Gripper (end-effector) frame  

### Environment Frames
- ${\mathcal{F}_{M_i}, i \in \{1,2,3,4\}}$: Fiducial marker frames  
- ${\mathcal{F}_{B}}$: Chessboard origin frame  

---

## SE(3) Transform Notation

A rigid-body transform from frame ${\mathcal{F}_A }$ to ${\mathcal{F}_B }$ is written as

$${
    ^WT_{B_O} = \begin{bmatrix} {^W\mathbf{R}_{B_O}} & {^W\mathbf{p}_{B_O}} \\\ 0 & 1 \end{bmatrix} = {^W\mathbf{T}_{C}} {^C\mathbf{T}_{B_O}} \
    {^C\mathbf{T}_{B_O}} = {^C\mathbf{T}_{M_i}} \space {^{M_i}\mathbf{T}_{B_O}} \qquad 1 \le i \le 4 \
    \\
}$$

where  
$${
    {}^W\mathbf{R}_{B_O} \in \mathrm{SO}(3) 
}$$ is a rotation matrix and,  
$${
    {}^W\mathbf{p}_{B_O} \in \mathbb{R}^3 
}$$ is a translation vector.

---

## Transform Chain

The pose of the chessboard in the world frame is obtained via composition in
${\mathrm{SE}(3)}$:

$${
    {}^{W}\mathbf{T}_{B}={}^{W}\mathbf{T}_{C}\,{}^{C}\mathbf{T}_{B}
}$$

The camera-to-board transform is computed from the detected fiducial markers:

$${
    {}^{C}\mathbf{T}_{B}={}^{C}\mathbf{T}_{M_i}\,{}^{M_i}\mathbf{T}_{B},\qquad i \in \{1,2,3,4\}.
}$$

---

## Board Origin Offset

The chessboard origin is defined relative to marker ${M_1 }$ by a fixed translation

$${
    {}^{M_1}\mathbf{p}_{B}=\begin{bmatrix}t \\\ t \\\ 0 \end{bmatrix},
}$$

yielding the homogeneous transform

$${
    {}^{M_1}\mathbf{T}_{B}=\begin{bmatrix} \mathbf{I}_{3 \times 3} & \begin{bmatrix} t \\\ t \\\ 0 \end{bmatrix} \\\ \mathbf{0}^{\mathsf{T}} & 1 \end{bmatrix}.
}$$