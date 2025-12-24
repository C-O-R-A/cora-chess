## Transforms

![Triads](assets/Transforms.png)

We define the following coordinate frames:

#### Robot Frames
- ${\mathcal{F}_W }$: World (robot base) frame  
- ${\mathcal{F}_C}$ : Camera frame  
- ${\mathcal{F}_G}$ : Gripper (end-effector) frame  

#### Environment Frames
- ${\mathcal{F}_{M_i}, i \in \{1,2,3,4\}}$: Fiducial marker frames  
- ${\mathcal{F}_{B_O}}$: Chessboard origin frame  

<!-- #### Notation

A rigid-body transform from frame ${\mathcal{F}_W }$ to ${\mathcal{F}_{B_O}}$ is written as

$${
    ^W{T}_{B_O} = \begin{bmatrix} {^W{\mathbf{R}}_{B_O}} & {^W{\mathbf{p}}_{B_O}} \\\ 0 & 1 \end{bmatrix}
}$$

where  
$${
    {}^W{\mathbf{R}}_{B_O} \in \mathrm{SO}(3) 
}$$ is a rotation matrix and,  
$${
    {}^W{\mathbf{p}}_{B_O} \in \mathbb{R}^3 
}$$ is a translation vector. -->

---

## Transform Chains

#### Camera to Board Origin
The pose of the chessboard in the world frame is obtained via composition in
${\mathrm{SE}(3)}$:

$${
    {}^{W}\mathbf{T}_{B_O}={}^{W}\mathbf{T}_{C}\,{}^{C}\mathbf{T}_{B_O}
}$$

The camera-to-board transform is computed from the detected fiducial markers:

$${
    {}^{C}\mathbf{T}_{B_O}={}^{C}\mathbf{T}_{M_i}\,{}^{M_i}\mathbf{T}_{B_O},\qquad i \in \{1,2,3,4\}.
}$$

#### Marker Offset

The chessboard origin is defined relative to marker ${M_1 }$ by a fixed translation

$${
    {}^{M_1}\mathbf{p}_{B_O}=\begin{bmatrix}t \\\ t \\\ 0 \end{bmatrix},
}$$

yielding the homogeneous transform

$${
    {}^{M_1}\mathbf{T}_{B_O}=\begin{bmatrix} \mathbf{I}_{3 \times 3} & \begin{bmatrix} t \\\ t \\\ 0 \end{bmatrix} \\\ \mathbf{0}^{\mathsf{T}} & 1 \end{bmatrix}.
}$$

#### Static Gripper Transform