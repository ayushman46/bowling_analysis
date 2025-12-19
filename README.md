# Cricket Bowling 3D Analysis App

An end-to-end application that analyzes cricket bowling videos using MediaPipe Pose and SPIN 3D reconstruction to compute biomechanics metrics.

<img width="661" height="393" alt="Image" src="https://github.com/user-attachments/assets/6e20f1c4-a4ef-4fc7-8136-d7d8a26dd9a0" />
<img width="713" height="393" alt="Image" src="https://github.com/user-attachments/assets/a9c11e00-936c-484b-8a6b-844b026eabe1" />

## Features

- **Video Processing**: Upload bowling videos via Streamlit interface
- **Delivery Detection**: Automatically detect the ball release frame using wrist velocity analysis
- **2D Pose Estimation**: MediaPipe Pose for robust 2D skeleton tracking
- **3D Reconstruction**: SPIN model for accurate 3D body mesh and joint positions
- **Biomechanics Analysis**: Compute key bowling metrics:
  - Elbow angle
  - Spine tilt
  - Shoulder abduction
  - Hip-shoulder separation
  - Release wrist height
- **Visualization**: View 2D skeleton overlays and 3D mesh renders
- **Export**: Download mesh (.obj), joints (.npy), metrics (.json), and frames (.jpg)

## Prerequisites

- Python 3.8+
- Virtual environment activated (`.venv`)
- SPIN repository cloned into `spin_src/`
- SMPL model files (see setup below)

## Project Structure

```
bowlin/
├── .venv/                      # Python virtual environment
├── spin_src/                   # SPIN repository (nkolot/SPIN)
│   ├── data/
│   │   ├── smpl/              # SMPL model files
│   │   │   ├── SMPL_NEUTRAL.pkl
│   │   │   ├── SMPL_MALE.pkl
│   │   │   └── SMPL_FEMALE.pkl
│   │   ├── model_checkpoint.pt
│   │   └── smpl_mean_params.npz
│   ├── models/
│   ├── utils/
│   └── ...
├── src/
│   ├── ingestion/             # Video loading and frame extraction
│   ├── pose2d/                # MediaPipe pose estimation
│   ├── delivery/              # Delivery frame detection
│   ├── reconstruction/        # SPIN wrapper
│   ├── analysis/              # Biomechanics metrics
│   └── visualization/         # Rendering utilities
├── results/                   # Output directory (created at runtime)
├── app.py                     # Streamlit application
├── config.py                  # Configuration
├── requirements.txt           # Python dependencies
├── setup_spin.sh             # SPIN data download script
└── README.md                  # This file
```

## Setup Instructions

### 1. Install Dependencies

First, ensure your virtual environment is activated:

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

Install main project dependencies:

```bash
pip install -r requirements.txt
```

Install SPIN-specific dependencies:

```bash
# Note: SPIN was designed for older Python/PyTorch versions
# We install only what's needed for inference with compatible versions
pip install tensorboard
```

**Important:** Some packages in `spin_src/requirements.txt` are incompatible with Python 3.12 or only needed for training:
-  `chumpy` - Python 3.11+ incompatible, training-only
-  `neural-renderer-pytorch` - Build issues, training-only  
-  `torchgeometry` - Deprecated, training-only
-  `spacepy` - Dataset preprocessing only

**All compatibility issues have been resolved!** See `COMPATIBILITY_FIXES.md` for details.

**Note**: If you encounter issues with pyrender, follow the [pyrender installation guide](https://pyrender.readthedocs.io/en/latest/install/index.html).

### 2. Download SPIN Model Data

Run the setup script to download SPIN model checkpoint and data files:

```bash
chmod +x setup_spin.sh
./setup_spin.sh
```

This will download:
- SPIN model checkpoint (~300MB)
- SMPL mean parameters and other data files (~200MB)

### 3. Get SMPL Model Files

The SMPL body model files cannot be automatically downloaded due to licensing restrictions. You need to:

1. **Download the neutral SMPL model**:
   - Go to http://smplify.is.tue.mpg.de
   - Register and download the neutral model
   - Rename to `SMPL_NEUTRAL.pkl`

2. **Download male and female models** (optional, for evaluation):
   - Go to http://smpl.is.tue.mpg.de
   - Register and download both models
   - Rename to `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`

3. **Place all files** in `spin_src/data/smpl/`:
   ```
   spin_src/data/smpl/
   ├── SMPL_NEUTRAL.pkl
   ├── SMPL_MALE.pkl
   └── SMPL_FEMALE.pkl
   ```

**Note**: The user indicated they already have these files. Verify they're in the correct location.

### 4. Verify Setup

Check that all required files are present:

```bash
ls -lh spin_src/data/model_checkpoint.pt
ls -lh spin_src/data/smpl_mean_params.npz
ls -lh spin_src/data/smpl/*.pkl
```

You should see:
- `model_checkpoint.pt` (~300MB)
- `smpl_mean_params.npz` and other .npz/.npy files
- Three SMPL .pkl files in the smpl/ directory

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser (typically at http://localhost:8501).

## Deploying on Streamlit Cloud

This repo is set up to deploy on Streamlit Cloud.

### Streamlit Cloud settings

- **App entrypoint**: `app.py`
- **Python runtime**: pinned via `runtime.txt` (currently `python-3.12`)
- **System packages**: `packages.txt` installs OS libs needed by OpenCV/MediaPipe on Linux

### Why you saw `np.float = np.float_` errors

On Streamlit Cloud you may get a newer Python/NumPy combo (e.g. Python 3.13 + NumPy 2.x).
NumPy 2.x removes legacy aliases like `np.float`, and accessing older aliases like
`np.float_` can error in some builds.

This app imports `chumpy_compat` first (see the top of `app.py`) to patch these
aliases safely.

### Common deployment gotchas

1. **Large model files**: SPIN checkpoints and SMPL files can be too large for the repo.
  - Keep them out of Git; download at runtime or store in a release bucket.
  - Make sure the app can start even if the model files are missing (show a friendly error).

2. **MediaPipe wheels**: Streamlit Cloud is Linux; MediaPipe works best when the Python
  version matches available wheels.
  - This repo pins **Python 3.12** in `runtime.txt`.

3. **GPU**: Streamlit Cloud is typically CPU-only.
  - Expect slower inference.


## Usage

1. **Upload Video**: Click "Upload bowling video" and select a cricket bowling video (MP4, MOV, or AVI)
   - Best results with side-view videos
   - Person should be relatively centered in frame

2. **Process**: Click the "Process Video" button

3. **Wait**: Processing typically takes 1-3 minutes depending on video length and hardware
   - The app will show progress for each step:
     - Frame extraction
     - 2D pose estimation
     - Delivery frame detection
     - 3D reconstruction
     - Metrics computation
     - Visualization

4. **Review Results**:
   - View the detected delivery frame with 2D skeleton overlay
   - See computed biomechanics metrics
   - View the 3D mesh preview

5. **Download**:
   - Click download buttons to save:
     - `mesh.obj` - 3D mesh file (can be opened in Blender, MeshLab, etc.)
     - `joints_3d.npy` - NumPy array of 3D joint positions
     - `metrics.json` - Computed biomechanics metrics
     - `delivery_frame.jpg` - The detected delivery frame image

## Understanding the Metrics

- **Elbow Angle**: Angle at the bowling arm elbow (degrees)
  - Lower values = more bent elbow
  - ~150-180° is typical for straight-arm bowling

- **Spine Tilt**: Forward lean of the spine (degrees)
  - Measured from vertical
  - Higher values = more forward lean

- **Shoulder Abduction**: Angle between torso and upper arm (degrees)
  - Measures how far the arm is raised from the body

- **Hip-Shoulder Separation**: Rotational difference between hips and shoulders (degrees)
  - Indicates trunk rotation during delivery

- **Release Wrist Height**: Vertical position of wrist at release (model units)
  - Higher values = higher release point

## Troubleshooting

### SPIN Model Errors

**Error**: `FileNotFoundError: SPIN checkpoint not found`
- **Solution**: Run `./setup_spin.sh` to download the checkpoint

**Error**: `SMPL model files not found`
- **Solution**: Download SMPL models from the official sources (see Setup section)

**Error**: `Could not import SPIN modules`
- **Solution**: Ensure `spin_src/` contains the official SPIN repository
- Verify by checking `spin_src/models/hmr.py` exists

### MediaPipe Errors

**Error**: `No pose detected in any frame`
- **Solution**: 
  - Ensure the person is clearly visible in the video
  - Try videos with better lighting/contrast
  - Person should be relatively centered

**Error**: `Not enough wrist points to detect delivery frame`
- **Solution**:
  - Video may be too short or person not visible throughout
  - Ensure the bowling action is complete in the video

### Rendering Errors

**Error**: Pyrender/OpenGL errors
- **Solution**:
  - On Linux: Ensure you have proper OpenGL support
  - Try setting `PYOPENGL_PLATFORM=egl` or `osmesa`
  - Install mesa libraries: `sudo apt-get install libosmesa6-dev`

### Memory Issues

**Error**: Out of memory errors
- **Solution**:
  - Use shorter videos
  - Reduce video resolution before upload
  - Close other applications

## Technical Details

### Pipeline Overview

1. **Ingestion** (`src/ingestion/video_utils.py`):
   - Load video with OpenCV
   - Extract frames and metadata

2. **2D Pose** (`src/pose2d/mediapipe_runner.py`):
   - Run MediaPipe Pose on all frames
   - Extract landmark positions in pixel space

3. **Delivery Detection** (`src/delivery/delivery_detector.py`):
   - Compute wrist trajectory from 2D poses
   - Find peak velocity (delivery moment)
   - Select sharpest frame in window around peak

4. **3D Reconstruction** (`src/reconstruction/spin_wrapper.py`):
   - Preprocess delivery frame (crop, resize, normalize)
   - Run SPIN HMR model
   - Generate SMPL mesh and joints

5. **Analysis** (`src/analysis/metrics.py`):
   - Compute joint angles from 3D positions
   - Calculate biomechanics metrics

6. **Visualization** (`src/visualization/render_utils.py`):
   - Draw 2D skeleton overlay
   - Render 3D mesh with pyrender

### Output Files

Each processing session creates a folder in `results/session_<uuid>/`:

```
results/session_<uuid>/
├── input_video.mp4                 # Original uploaded video
├── frames/                         # Extracted frames
├── keypoints_2d.json              # 2D pose data
├── delivery_frame_info.json       # Detection metadata
├── delivery_frame.jpg             # Detected frame
├── delivery_frame_with_skeleton.jpg  # Frame with overlay
├── mesh.obj                       # 3D mesh
├── joints_3d.npy                  # 3D joint positions
├── smpl_params.npz                # SMPL model parameters
├── metrics.json                   # Biomechanics metrics
└── mesh_preview.jpg               # Rendered mesh image
```

## Credits

This application integrates:

- **SPIN** (Kolotouros et al., ICCV 2019): 3D human pose and shape reconstruction
  - Repository: https://github.com/nkolot/SPIN
  - Paper: https://arxiv.org/abs/1909.12828

- **MediaPipe** (Google): Real-time pose estimation
  - https://google.github.io/mediapipe/

- **SMPL** (Loper et al., 2015): Skinned multi-person linear model
  - https://smpl.is.tue.mpg.de/

## License

This project integrates multiple components with different licenses:
- SPIN: See spin_src/LICENSE
- SMPL: Requires registration and agreement to license terms
- MediaPipe: Apache 2.0

Please respect the licenses of all integrated components.

## Citation

If you use this tool in research, please cite the relevant papers:

```bibtex
@Inproceedings{kolotouros2019spin,
  Title          = {Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop},
  Author         = {Kolotouros, Nikos and Pavlakos, Georgios and Black, Michael J and Daniilidis, Kostas},
  Booktitle      = {ICCV},
  Year           = {2019}
}
```
