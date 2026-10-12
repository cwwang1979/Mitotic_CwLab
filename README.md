# Mitotic_CwLab

**Mitotic_CwLab** is a deep-learning method for detecting and examining mitotic figures in histopathological whole-slide images (WSIs). The repository covers WSI preprocessing, patch-level annotation conversion, YOLO-based training, inference, reconstruction of whole-slide predictions, and quantitative evaluation.

The workflow uses **1024 × 1024** image patches and five annotation categories. Its principal target for mitotic figure detection is **T2 (Mitotic figure)**.

**Repository:** https://github.com/cwwang1979/Mitotic_CwLab


## Associated Publications

If you use this work or any part of this repository in your research, please cite:

- (**Under Revision**) **Wang et al. (2026).** *Improved Deep Learning for Mitotic Figure Examination on Whole Slide Images*. , *Biomedical Signal Processing and Control*.

A complete bibliographic citation will be added when the paper becomes available.

### Download

Clone the repository:

```bash
git clone https://github.com/cwwang1979/Mitotic_CwLab.git
cd Mitotic_CwLab
```

Data, execution files, configuration files, and pretrained models can be downloaded from the [zip](https://drive.google.com/file/d/1K_UESR6lhJi4tFSrzTYq6w8prpMZqRPB/view?usp=drive_link) file.


## Demonstration Video

**Cwlab AiExplore — Interactive Mitotic Figure Detection on Whole Slide Images**

[Watch the demonstration video on YouTube](https://www.youtube.com/watch?v=8oiq_sFF2uw)

This video demonstrates **Cwlab AiExplore**, an in-house platform integrating digital pathology image viewing with deep learning inference for real-time mitotic figure detection in whole-slide images (WSIs). Users can select a region of interest (ROI), initiate AI inference, and inspect detection overlays directly on the original pathology image. The demonstration covers the workflow from ROI selection to inference and result visualization, including analysis across multiple selected regions.

**Features demonstrated:**

- Interactive whole-slide image viewing.
- User-defined region-of-interest (ROI) selection.
- Real-time AI inference for mitotic figure detection.
- Detection overlays for visual inspection.
- Inference across multiple selected ROIs.

By bringing image viewing and AI inference into a unified interface, Cwlab AiExplore supports interactive exploration of model predictions for digital pathology research.

**Developed by Cwlab.**

**Related manuscript:** *Improved Deep Learning for Mitotic Figure Examination on Whole Slide Images* (under revision, *Biomedical Signal Processing and Control*).

## Setup

### Requirements

The following hardware and software specifications follow the reference environment supplied for this project.

| Component | Specification |
| --- | --- |
| Operating system | Ubuntu 18.04 |
| GPU memory | ≥ 16 GB |
| NVIDIA GPU driver | ≥ 530.30.02 |
| CUDA | ≥ 12.1 |
| Python | 3.8.20 |
| PyTorch | 2.4.1 (CUDA 12.1 build) |
| torchvision | 0.19.1 |
| opencv-python | 4.11.0.86 |
| Ultralytics | 8.3.27 |

WSI extraction also requires **OpenSlide**. The evaluation scripts use NumPy, pandas, scikit-learn, and other scientific Python packages.


### 1. Installation

Create and activate the Conda environment:

```bash
conda create -n Mitotic_CwLab python=3.8.20 -y
conda activate Mitotic_CwLab
```

Install a matching PyTorch/CUDA build:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the repository's pinned Ultralytics version and additional packages used by the supplied scripts:

```bash
pip install -r requirements.txt
pip install opencv-python==4.11.0.86 openslide-python Pillow pandas scikit-learn matplotlib
```

On Ubuntu, the OpenSlide system library may also be needed:

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 openslide-tools
```

Verify the environment:

```bash
python -c "import torch, ultralytics, cv2, openslide; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__)"
```

> **Note:** The supplied `requirements.txt` currently pins `ultralytics==8.3.27`. The other versions above are documented from the reference setup and installation instructions, rather than being additional entries in that file.

## Data Format

### 2. Original whole-slide dataset

Organize the original WSIs and their annotations as follows:

```text
Mitotic_CwLab/
├── MITO_image/
│   ├── train/
│   │   ├── <slide_01>.svs
│   │   └── ...
│   └── test/
│       ├── <slide_22>.svs
│       └── ...
├── MITO_anno/
│   ├── train/
│   │   ├── <slide_01>.svs/
│   │   │   └── ntuhong2.json
│   │   └── ...
│   └── test/
│       ├── <slide_22>.svs/
│       │   └── ntuhong2.json
│       └── ...
└── ...
```

The research split comprises **21 training WSIs** and **11 independent test WSIs**. The subsequent training/validation subdivision operates on patches drawn from the training WSIs, not on additional independent WSIs.

The `ntuhong2.json` files contain an `annotations` array with fields including `DiseaseType`, `Points`, and bounding-box coordinates (`x_min`, `y_min`, `x_max`, `y_max`).

### Annotation categories

| YOLO class ID | Annotation label | Category |
| --- | --- | --- |
| 0 | T1 | Granulocyte |
| 1 | T2 | Mitotic figure |
| 2 | T3 | Tumor cell |
| 3 | T4 | Mitotic figure lookalike |
| 4 | T7 | Other ambiguous cells |

A YOLO detection annotation is one line per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All four box coordinates are normalized to the range **[0, 1]**.

## Steps

### 3. Training WSI preprocessing

Use **`patch_train.py`** to extract annotated patches from training WSIs. In its current implementation:

- Patch size: **1024 × 1024** pixels.
- Sliding-window stride: **1024** pixels (no overlap between regularly spaced patches).
- OpenSlide resolution level: **0**.
- Partial edge patches are padded to 1024 × 1024.
- **Only patches containing at least one intersecting annotation are saved.**
- Bounding-box annotation coordinates are transformed from WSI space to patch space.

Before running, configure the paths at the end of `patch_train.py` for the **training** set:

```python
image_dir = "./MITO_image/train"
train_dir = "./MITO_anno/train"
patch_image_dir = "train_patch_image"
patch_anno_dir = "train_patch_anno"
```

Then run:

```bash
python patch_train.py
```

Typical output:

```text
train_patch_image/
├── <slide_01>.svs_0_0.png
└── ...

train_patch_anno/
├── <slide_01>.svs_0_0/
│   └── ntuhong2.json
└── ...
```

The `_x_y` suffix records the patch's upper-left location in the original WSI and is required when restoring predictions to whole-slide coordinates.

### 4. Convert training annotations to YOLO format

Use **`patch_json2txt.py`** to convert patch annotation JSON files into YOLO label files. Before running, set the script's input and output directories to:

```python
patch_anno_dir = "./train_patch_anno"
output_dir = "./yolo_train_patch_anno"
```

Run:

```bash
python patch_json2txt.py
```

Each patch annotation is converted to a same-named `.txt` file with normalized YOLO boxes and class IDs.

> The separate `json2txt.py` utility processes its own `data.json`/`ntuhong2.json` folder structure; it is **not** the patch-label conversion step described here.

### 5. Split training and validation patches

Use **`data_split.py`** together with **`annotations_summary_expanded.csv`**:

```bash
python data_split.py
```

The supplied script reads:

```text
train_patch_image/
yolo_train_patch_anno/
annotations_summary_expanded.csv
```

It groups candidate images by the `Class_*` columns in the CSV, processes less frequent classes first, and allocates approximately **87.5% / 12.5%** of candidates in each pass to training and validation, respectively.

The script writes the following directory structure:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── <patch_01>.png
│   │   └── ...
│   └── labels/
│       ├── <patch_01>.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

> **Important:** This is a **patch-level** validation split. The provided `annotations_summary_expanded.csv` also contains a final `Total` summary row; exclude that row when constructing candidate image records in a cleaned version of the split script. Retain the independent test WSIs outside both training and validation.

### 6. Configure the YOLO dataset

The latest supplied `data_path.yaml` contains an absolute path from the original workstation and assumes `images/train` and `images/val`. Since `data_split.py` actually writes `dataset/train/images` and `dataset/val/images`, update the YAML for the generated directories before training.

For the structure above, an appropriate configuration is:

```yaml
path: /absolute/path/to/Mitotic_CwLab/dataset
train: train/images
val: val/images
nc: 5
names:
  - Granulocyte
  - Mitotic_figure
  - Tumor_cell
  - Mitotic_figure_lookalike
  - Other_ambigous_cells
```

Keep the annotation and class order consistent with the `patch_json2txt.py` mapping.

## Training

### 7. Fine-tune YOLO11x

The **latest supplied `train.py`** loads the pretrained **`yolo11x.pt`** checkpoint and fine-tunes it with these settings:

| Parameter | Value |
| --- | --- |
| Base weights | `yolo11x.pt` |
| Epochs | `500` |
| Image size | `1024` |
| Batch size | `4` |
| GPU | `0` |
| Mosaic | Enabled |
| MixUp | `0.5` |
| Augmentation | Enabled |
| Output project | `trained_model` |
| Training run | `YOLOv11x_500_1024_4_mosaic1_mixup05` |

The parameters are defined **inside the Python script**; this version does not implement a custom training command-line argument parser.

```bash
python train.py
```

After training, the weights are saved under:

```text
trained_model/
└── YOLOv11x_500_1024_4_mosaic1_mixup05/
    └── weights/
        ├── best.pt
        └── last.pt
```


## Inference

### 8. Extract all test WSI patches

Use **`patch_inf.py`** for test preprocessing:

```bash
python patch_inf.py
```

The script reads `./MITO_image/test` and `./MITO_anno/test`, and writes:

```text
test_patch_image/
test_patch_anno/
```

Unlike the training patch script, **`patch_inf.py` saves every sliding-window patch**, including patches without annotations. Empty patch annotation JSON files are saved as well. This is important for whole-slide evaluation because predictions from negative/background areas can contribute false positives.

The patch size, stride, resolution level, and edge-padding behavior are the same as described for training.

### 9. Run model prediction

The supplied **`val.py`** performs prediction by loading a `best.pt` checkpoint. For a selected training run, set the following values consistently in `val.py`:

```python
modelname = "YOLOv11x_500_1024_4_mosaic1_mixup05"  # example: standard YOLO11x run

model = YOLO("./trained_model/" + modelname + "/weights/best.pt")

for result in model.predict(
    source="./test_patch_image",
    imgsz=1024,
    save=False,
    save_txt=True,
    save_crop=False,
    project="./MITOS_result",
    name=modelname,
    stream=True,
):
    pass
```

This excerpt shows the **path and run-name changes required to align the existing prediction script with the earlier steps**; it is not a separate `inference.py` CLI implementation. You may instead use your own GAM run name and checkpoint if that model is available.

Run:

```bash
python val.py
```

Prediction labels will be written to:

```text
MITOS_result/
└── <modelname>/
    └── labels/
        ├── <slide_name>.svs_<x>_<y>.txt
        └── ...
```

With the supplied settings, output rows contain five columns:

```text
<class_id> <x_center> <y_center> <width> <height>
```

The `conf` argument is not set explicitly in the latest supplied `val.py`; the Ultralytics prediction default applies. If you enable `save_conf=True`, the output gains a sixth confidence column, and `txt2json.py` must be updated to parse it.

## Post-processing

### 10. Convert prediction TXT files to patch JSON

Run **`txt2json.py`** after updating its paths and `modelname` to match the inference run:

```python
modelname = "YOLOv11x_500_1024_4_mosaic1_mixup05"  # example
image_folder = "./test_patch_image"
label_folder = "./MITOS_result/" + modelname + "/labels"
json_output_folder = "./MITOS_result_cloudfarmat_patch/" + modelname
```

```bash
python txt2json.py
```

This utility converts normalized YOLO boxes back into absolute patch coordinates and maps each YOLO class ID to its original `DiseaseType` label. It creates a project-specific JSON annotation file for each patch with predictions.

**Check the empty-prediction handling** in the supplied script before relying on it for a full WSI run; its final block references `yolo_path` before defining it and should be corrected. Also ensure that patches with no predictions receive an empty prediction JSON when needed.

### 11. Restore patch predictions to WSI coordinates

Use **`patchjson2wholesjson.py`**:

```bash
python patchjson2wholesjson.py
```

Before running, set its `modelname` to the same run name used for inference and TXT-to-JSON conversion.

The script:

1. Reads the source-slide identifier and `(x, y)` offset from each patch filename.
2. Adds the offsets to prediction bounding boxes and point coordinates.
3. Attempts to merge annotations associated with neighboring patch boundaries.
4. Saves one prediction JSON per WSI.

Output:

```text
MITOS_result_cloudfarmat_whole/
└── <modelname>/
    ├── <slide_01>.svs/
    │   └── <modelname>.json
    └── ...
```

> The supplied boundary-merging implementation is part of this project's original workflow. It should not be described as a general IoU-based global non-maximum-suppression implementation without additional changes or verification.

## Evaluation

### 12. Whole-slide quantitative evaluation

Use **`Cloud_Detec_evaluation.py`** to compare reconstructed WSI predictions with the original `ntuhong2.json` ground truth.

The evaluation script expects each slide directory to contain:

```text
MITOS_result_cloudfarmat_whole/
└── <modelname>/
    ├── <slide_01>.svs/
    │   ├── ntuhong2.json
    │   └── <modelname>.json
    └── ...
```

Copy or link the **original WSI-level** `ntuhong2.json` file into the matching output directory for each slide. Do **not** replace it with the patch-local annotation JSON. Then set the evaluation script's `root` and `model` variables to match the selected run and execute:

```bash
python Cloud_Detec_evaluation.py
```

The supplied evaluation code uses **IoU ≥ 0.5** as its geometric matching criterion. It calculates detection and classification metrics for multiple cell types and generates per-slide and aggregate results in **JSON**, **CSV**, and **NumPy (`.npy`)** formats.

**Reported evaluation results for the Proposed Method:** Only the **T2 (Mitotic figure)** row under **Detect** in the **`Total_result`** section of the output CSV is used. The corresponding metrics are **Accuracy, Recall, Precision, and F1-score (`FMeasure`)**. These values are aggregated across the test WSIs using the evaluator's total TP/FP/TN/FN counts, rather than taken from the `Average_result` section.

```text
Total_result
                 Detect
ClassNumber   Accuracy   Recall   Precision   FMeasure
T2            ...        ...      ...         ...
```

For the JSON output, the same metrics are stored under the `T2` item within the top-level `Total` array, in its `Detect` object.

Do **not** substitute the `Class` metrics, other cell types, the `Merge` entry, or the all-class `Total` entry for the reported Proposed Method results. **No micrometer-distance matching criterion is used for the Proposed Method evaluation.**

> The evaluator also produces additional categories (including T5/T6 and `Merge`) because `ClassNum = 7`; these are not part of the reported T2 detection result.


## License

This source code is provided under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** terms for non-commercial use, as stated in the reference project description. For commercial licensing, please contact Prof. Ching-Wei Wang.

License summary: https://creativecommons.org/licenses/by-nc/4.0/

## Contact

**Prof. Ching-Wei Wang**  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
National Taiwan University of Science and Technology
