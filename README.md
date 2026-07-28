# Embodied Semantic Navigation

Autonomous embodied navigation in AI2-THOR with an exploration stack that builds an occupancy grid, a semantic memory of observed objects, and a language-to-goal layer for natural-language navigation.

The project started as frontier-based exploration and grew into a pipeline that can:

* explore an unknown indoor scene,
* detect and record objects with YOLO,
* store those detections in a semantic map,
* resolve natural-language requests with CLIP,
* and navigate to the best known location of the requested object.

## Current Status

This repository is organized around the following phases:

* Phase 3: frontier exploration + occupancy-grid navigation
* Phase 4: perception wiring + semantic map construction
* Phase 5: CLIP-based language goal resolution + goal-directed navigation
* Phase 6: polish, demo capture, and portfolio presentation

## Pipeline

flowchart LR
	A["User query\n'Find something I can drink from'"] --> B[CLIP text encoding]
	B --> C[Semantic label ranking]
	C --> D["Best label\n'Mug'"]
	D --> E[SemanticMap lookup]
	E --> F["Best known cell\nor nearest matching cell"]
	F --> G[Navigator BFS]
	G --> H[AI2-THOR movement]
	H --> I[Goal reached / target confirmed]

	J[AI2-THOR frame stream] --> K[YOLOv8 detector]
	K --> L[SemanticMap update]
	L --> E

	M[Frontier explorer] --> N[Occupancy grid]
	N --> G


## Repository Layout

```text
embodied-semantic-nav/
├── explore.py              # exploration demo + semantic mapping run
├── goal_seek.py            # end-to-end language goal demo
├── navigation/
│   ├── frontier.py         # frontier detection
│   ├── navigator.py        # movement, BFS, exploration
│   └── occupancy_grid.py   # world/grid representation
├── mapping/
│   └── semantic_map.py     # grid-based semantic memory
├── perception/
│   └── perception.py       # YOLO detector and label mapping
├── language/
│   ├── clip_resolver.py    # CLIP query-to-label resolution
│   └── goal_navigator.py   # two-stage goal execution
├── tests/
│   └── test_semantic_map.py
└── README.md
```

## What It Does

1. Explore an AI2-THOR scene using frontier-based navigation.
2. Update an occupancy grid from motion and visibility.
3. Run YOLO on each frame and add detections to a semantic map.
4. Use CLIP to match a natural-language query to a semantic label.
5. Find the best known cell for that label, or explore until it is observed.
6. Navigate to the target cell and confirm the object in view.

## Example Use

The primary demo scripts are:

* `explore.py` for exploration + semantic mapping
* `goal_seek.py` for the full language-to-navigation pipeline

Example goal queries:

* `a refrigerator`
* `something I can drink from`
* `something used to cook food`
* `a household appliance`

## Setup

This project assumes Python 3.10+ and AI2-THOR. It has been developed on macOS with a local virtual environment, but the code path is simulator-agnostic as long as the dependencies are available.

```bash
git clone <your-repo-url>
cd embodied-semantic-nav

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install ai2thor ultralytics torch torchvision numpy matplotlib opencv-python pillow pytest
pip install git+https://github.com/openai/CLIP.git
```

Notes:

* `CLIP` is installed from the OpenAI repository and used by `language/clip_resolver.py`.
* YOLO weights are loaded from `perception/yolov8s.pt`.
* The first AI2-THOR launch may download scene assets.

## Run

Explore and build the semantic map:

```bash
python explore.py
```

Run the full natural-language goal pipeline:

```bash
python goal_seek.py
```

## Results Snapshot

The current implementation already produces a semantic map and a goal-resolution pipeline. For the final public release, I recommend adding a small quantitative table here with:

* CLIP label-resolution accuracy across 10+ queries
* navigation success rate on held-out target objects
* map coverage or number of unique cells explored
* average steps per successful goal

## Limitations

This is a practical robotics prototype, not a full SLAM system.

Known limitations:

* The semantic map is built from detector observations, so it depends on what the agent has already seen.
* CLIP currently resolves against a candidate label set rather than unrestricted open-vocabulary entities.
* Navigation is discrete and grid-based, so it is not optimal in continuous space.
* Detector noise can still introduce false positives or semantically ambiguous labels.

## What To Add Next

Phase 6 should focus on public-facing polish:

* record a demo video with frame capture
* add a concise architecture diagram
* include the results table from evaluation runs
* add a short section explaining design choices and failure cases

## Suggested License

If you plan to publish this openly, add a license file before sharing the repository publicly.
