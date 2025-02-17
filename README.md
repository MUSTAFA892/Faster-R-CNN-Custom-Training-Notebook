
---

# Vehicle Detection with Faster R-CNN

This repository demonstrates how to use the Faster R-CNN model to detect vehicles (cars) in videos using PyTorch and Torchvision. The project includes scripts for both training a custom model and performing inference on videos to detect vehicles.

## Table of Contents
- [Overview](#overview)
- [Faster R-CNN Explanation](#faster-r-cnn-explanation)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)
- [License](#license)

## Overview

Faster R-CNN (Region-based Convolutional Neural Networks) is a deep learning model used for object detection. This repository provides an implementation where Faster R-CNN is used to detect vehicles in video files. The project includes code for training the model on a custom vehicle detection dataset, running inference on videos, and saving processed output videos with bounding boxes around detected vehicles.

## Faster R-CNN Explanation

Faster R-CNN is a two-stage object detection framework that uses a Region Proposal Network (RPN) to propose candidate object regions and a Fast R-CNN detector to classify and refine the regions. It is one of the most widely used models for object detection due to its high accuracy and speed.

- **Stage 1: Region Proposal Network (RPN)**: The RPN generates region proposals that may contain objects by sliding over the image and producing bounding boxes with objectness scores.
- **Stage 2: Object Detection**: The generated region proposals are then passed through a Fast R-CNN network that classifies the objects and refines the bounding boxes.

The Faster R-CNN model used here is built on the ResNet-50 backbone with a Feature Pyramid Network (FPN) to enhance multi-scale object detection. It is trained on a custom dataset of vehicle images.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/faster-rcnn-vehicle-detection.git
   cd faster-rcnn-vehicle-detection
   ```

2. Set up a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) If you don’t have the model weights, make sure to follow the instructions for training the model on your own dataset.

## Dataset Structure

The dataset for training should be organized as follows. A **single JSON file** will store the annotations for all images in the respective training and validation sets. Each entry in the JSON file will represent an image along with the annotations (bounding boxes and labels) for that image.

```
dataset/
│
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations/
│       └── train_annotations.json
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations/
│       └── val_annotations.json
```

### JSON Annotation Format

The annotations are stored in a single JSON file for each set (`train_annotations.json` and `val_annotations.json`). The JSON file contains the following structure:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    },
    {
      "id": 2,
      "file_name": "image2.jpg",
      "width": 640,
      "height": 480
    },
    ...
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 1,
      "bbox": [x_min, y_min, width, height],
      "area": width * height,
      "iscrowd": 0
    },
    {
      "image_id": 2,
      "category_id": 1,
      "bbox": [x_min, y_min, width, height],
      "area": width * height,
      "iscrowd": 0
    },
    ...
  ],
  "categories": [
    {
      "id": 1,
      "name": "car"
    }
  ]
}
```

- **images**: A list of images, each with an `id`, `file_name`, `width`, and `height`.
- **annotations**: A list of bounding box annotations for each image. The `bbox` is defined by `[x_min, y_min, width, height]` and the `category_id` refers to the object class (in this case, "car").
- **categories**: The object classes, with an ID and name.

## Training the Model

1. **Prepare Your Dataset**:
   Ensure your dataset is organized as described above. Place your images in the `images/` folder and annotations in the corresponding JSON files (`train_annotations.json`, `val_annotations.json`).

2. **Run the Training Script**:
   The `train.py` script is used to train the Faster R-CNN model on the custom dataset. You can start training with the following command:
   
   ```bash
   python train.py --dataset_path /path/to/dataset --epochs 10 --batch_size 4 --lr 0.005
   ```

   Replace `/path/to/dataset` with the actual path to your dataset folder.

3. **Monitor Training**:
   The training script will output logs to the console. You can adjust the number of epochs, learning rate, and batch size according to your requirements.

4. **Save the Model**:
   Once training is completed, the model's weights will be saved to a `.pth` file, which can be used for inference.

## Inference

Once the model is trained, you can run inference on a video using the following script:

1. **Run Inference on a Video**:
   To perform inference on a video and save the output with vehicle detection, run:

   ```bash
   python inference.py --video_path /path/to/video --output_path /path/to/output_video --model_path /path/to/trained_model.pth
   ```

   Replace `/path/to/video`, `/path/to/output_video`, and `/path/to/trained_model.pth` with the appropriate paths.

2. **View Output Video**:
   The output video will be saved as an MP4 file with bounding boxes drawn around detected vehicles.

## Results

After running inference on a video, the model will output a processed video file with the detected vehicles highlighted by bounding boxes. The results will be stored in the location specified in the `--output_path` parameter.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Requirements (`requirements.txt`)

Here are the dependencies you need for this project:

```
torch
torchvision
torchaudio
pycocotools
jupyter
notebook
opencv-python
matplotlib
PIL
```

To install these dependencies, you can run:

```bash
pip install -r requirements.txt
```

---

