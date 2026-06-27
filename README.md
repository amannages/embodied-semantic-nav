# Semantic Indoor Navigation for Embodied Household Assistants

[![Python](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://allenai.org)
[![AI](https://shields.io)](https://github.com)

An autonomous navigation system for embodied robotic agents inside simulated household environments. This project combines deep computer vision models with topological mapping to enable robots to locate, navigate to, and interact with objects using natural semantic commands.

---

## 📌 Project Overview

Traditional robotic navigation relies heavily on geometric maps (point clouds and grids) which lack semantic understanding of the world. This project implements **Semantic Mapping and Navigation**, allowing an embodied assistant to understand *what* objects are and *where* they sit in context. 

Instead of moving blindly to coordinates, the agent utilizes a visual-language approach to understand commands like *"find the coffee mug on the kitchen counter"* and independently plans an optimal path using simulated sensor data.

### Key Features
*   **Active Perception:** Real-time object detection using YOLOv8 and open-vocabulary scene understanding using CLIP.
*   **Simulated Embodiment:** High-fidelity physics and photo-realistic interactions powered by the AI2-THOR simulator.
*   **Semantic Mapping:** Graph-based topological mapping (NetworkX) tracking spatial relationships between household objects.
*   **Path Planning:** Intelligent pathing utilizing a mixture of frontier exploration and semantic visual cues.

---

## 🏗️ Project Architecture

```text
├── config/             # Environment and model hyperparameters
├── data/               # Saved maps, trajectories, and evaluation logs
├── models/             # Custom network definitions, YOLO weights, and CLIP hooks
├── src/
│   ├── controller.py   # AI2-THOR environment wrappers and agent API
│   ├── mapping.py      # Semantic and topological graph generation
│   ├── perception.py   # Object detection and visual feature extraction
│   └── planning.py     # Navigation algorithms and heuristic policies
├── tests/              # Unit testing for environment and models
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🚀 Phase 0: Environment Setup

### 1. Prerequisites
Ensure you have [Conda](https://conda.io) installed. This setup is optimized for Apple Silicon (M1/M2/M3 chips) using the MPS backend, but works on standard CUDA/CPU configurations.

### 2. Installation
Clone the repository and build the dedicated environment:

```bash
# Clone the repository
git clone https://github.com
cd YOUR-REPO-NAME

# Create and activate environment
conda create -n embodied-nav python=3.10 -y
conda activate embodied-nav

# Install Core Dependencies
pip install torch torchvision
pip install ultralytics transformers ai2thor
pip install numpy matplotlib opencv-python networkx
```

### 3. Verification
Verify your simulator environment works by running the initialization test script:

```bash
python -c "from ai2thor.controller import Controller; controller = Controller(scene='FloorPlan1', width=300, height=300); event = controller.step(action='MoveAhead'); print('Agent Position:', event.metadata['agent']['position']); controller.stop()"
```
*Note: The first run auto-downloads kitchen assets (~2–3 minutes).*

---

## 📅 Development Roadmap

- [x] **Phase 0:** Environment Setup & Simulator Verification
- [ ] **Phase 1:** Active Perception & Object Detection (YOLOv8 Integration)
- [ ] **Phase 2:** Semantic Mapping & Topology Graph Construction
- [ ] **Phase 3:** Open-Vocabulary Goal Targeting (CLIP Integration)
- [ ] **Phase 4:** Navigation Policy Evaluation & Benchmark Reporting

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
