# Deepfake Detection with EfficientNet-B0

This project trains a deep learning model to classify face images as real or deepfake-generated. It uses a pretrained EfficientNet-B0 and fine-tunes it for binary classification.

Built this as part of a university assignment on fake news detection and mitigation. The idea is that a model like this could be plugged into a larger pipeline that flags suspicious images on social media before they spread.

## How it works

The model takes in a face image (224x224), runs it through EfficientNet-B0 with the last layer swapped out for a 2-class output (real or fake), and gives a prediction. Transfer learning does most of the heavy lifting here since EfficientNet already knows how to extract visual features from being trained on ImageNet.

## Dataset

I used the [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) dataset from Kaggle. It originally has around 140k images but I subsampled it to 4k per class per split because my GPU couldn't handle the full thing in a reasonable time.

The dataset is too large to include in this repo. Download it from Kaggle and structure it like this:

```
Dataset/
  Train/
    Real/
    Fake/
  Validation/
    Real/
    Fake/
  Test/
    Real/
    Fake/
```

There's a `trim_dataset.py` script to cut each folder down to 4000 images if you want to do the same thing I did.

## Results

- Test accuracy: ~91.3%
- Better at catching fakes (94.5% recall) than correctly identifying real images (88.2% recall)
- Some overfitting visible in training curves

![Training Curves](<img width="1200" height="500" alt="training_curves" src="https://github.com/user-attachments/assets/6f3ca968-0fb3-47d0-b6b2-bc5b3fafc1f7" />)
![Confusion Matrix](<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/782854da-626c-4e33-8052-92cbbf22b437" />)
![Sample Predictions](<img width="1400" height="700" alt="sample_predictions" src="https://github.com/user-attachments/assets/1d0b59ca-cf8a-46f6-b407-78adc14be700" />)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib scikit-learn seaborn
```

Windows:
```bash
venv\Scripts\activate
```

## Running

Put the script in the same directory as your dataset folders and run:

```bash
python deepfake_detect.py
```

It'll train for 8 epochs and save three plots + the model weights (`deepfake_model.pth`).

Trained this on a GTX 1650 4GB, took about 15-20 minutes. Should work on anything with a CUDA GPU honestly. CPU would take a lot longer but it'd still work.
