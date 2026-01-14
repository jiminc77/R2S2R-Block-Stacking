# R2S2R Block Stacking

**Real-to-Sim-to-Real Block Stacking with UR5e Digital Twin**

A robust robotic manipulation system that stacks blocks on unstable towers using multi-view perception, Isaac Lab physics simulation, and domain randomization for reliable sim-to-real transfer.

![Demo](Demo.gif)

**[▶️ Watch Full Demo Video on YouTube](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)**

## Overview

This project demonstrates **sim-to-real transfer** for contact-rich manipulation tasks. A UR5e collaborative robot safely stacks blocks onto unstable towers by:

1. **Precisely perceiving** existing block positions using multi-view ArUco marker detection + depth fusion (±2mm accuracy)
2. **Simulating thousands of placement candidates** in Isaac Lab (8184 parallel environments with domain randomization)
3. **Executing only verified placements** on the real robot (95%+ success rate)

**Key Innovation**: By validating placements in simulation before execution, we achieve high success rates on unstable configurations while minimizing risk to hardware.


## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    EXECUTION PIPELINE                      │
├────────────────────────────────────────────────────────────┤
│  Phase 1: Multi-View Scanning                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ RIGHT → CENTER → LEFT viewpoints                     │  │
│  │ ArUco detection + Depth refinement                   │  │
│  │ Weighted fusion → /block_poses                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                │
│  Phase 2: Policy Search (Isaac Lab - Docker)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Spawn blocks in 8184 parallel envs                   │  │
│  │ Generate candidates (grid + orientations)            │  │
│  │ Geometric filtering (support polygon check)          │  │
│  │ Physics verification with domain randomization       │  │
│  │ Select best placement → /optimal_stacking/result     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                │
│  Phase 3: Real Robot Execution                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ State Machine: APPROACH → GRASP → LIFT →             │  │
│  │                TRANSPORT → PLACE → RETREAT           │  │
│  │ MoveIt planning with collision avoidance             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘

┌──────────────────┐                      ┌──────────────────┐
│   CONTROL PC     │ ←── ROS1 Network ──→ │ VISION/SIM PC    │
│  (No GPU)        │                      │  (RTX 2080 Ti)   │
├──────────────────┤                      ├──────────────────┤
│ • UR5e Driver    │                      │ • RealSense SDK  │
│ • MoveIt         │                      │ • ArUco Detection│
│ • Gripper Control│                      │ • Isaac Lab (GPU)│
│ • Robot Exec     │                      │ • Policy Search  │
└──────────────────┘                      └──────────────────┘
```


## Quick Start

### 1. Launch Control PC Services
```bash
# On Control PC
roslaunch control_pc.launch
```
This starts:
- UR5e robot driver
- MoveIt move_group
- Robot motion action server
- Gripper action server
- Hand-eye TF publisher

### 2. Launch Vision PC Services
```bash
# On Vision PC
roslaunch vision_pc.launch
```

### 3. Run Full Stacking Pipeline
```bash
# On Vision PC
cd vision_sim_pc
python3 run_full_stacking_pipeline.py
```

This executes:
1. Multi-view scanning (ArUco detection + depth fusion)
2. Isaac Lab policy search (8184 parallel environments)
3. Waits for robot controller to execute placement

### 4. Execute Robot Placement (Optional - Auto-triggered)
```bash
# On Control PC
cd robot_control_pc/scripts
python3 run_robot_stacking.py
```


## Technical Details

### Multi-View Fusion Algorithm

**Scanning Protocol**:
- 3 viewpoints: RIGHT (45°), CENTER (top-down), LEFT (45°)
- 1.5s stabilization wait per viewpoint (eliminate robot vibration)
- 3s continuous scanning (~45 frames @ 15Hz)

**Fusion Pipeline**:
1. **Outlier Rejection**: Remove detections >15mm from 3D median
2. **Weighted Averaging**:
   - Distance-based weight: `w_dist = exp(-distance / scale)`
   - View priority: CENTER view gets 2x weight
3. **Rotation Clustering**: Group quaternions with dot product > 0.8

### Domain Randomization

**Position Noise**:
- XY plane: ±2mm standard deviation
- Z axis: ±3mm standard deviation (accounts for drop during placement)

**Orientation Noise**:
- Yaw (Z-rotation): ±5° standard deviation
- Pitch/Roll: ±2° standard deviation

**Velocity Perturbations**:
- Linear velocity: 0.01 m/s std dev
- Angular velocity: 0.05 rad/s std dev

### Two-Stage Placement Optimization

**Stage 1: Geometric Pre-filtering** (milliseconds)
- Generate ~500-1000 candidates (±25mm grid, 5mm spacing, 5 orientations)
- Support polygon check: Verify tower COM stays within base polygon
- Result: ~100-200 candidates remain

**Stage 2: Physics Verification** (tens of seconds)
- Test each candidate in 128 parallel environments
- Apply domain randomization per environment
- Simulate 60 timesteps (1 second physics rollout)
- Calculate success rate: blocks remain stable (Z change < 30mm)
- Select: Success rate > 80%, closest to tower COM
