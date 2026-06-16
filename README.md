# Face Mask Detection Project

An end-to-end deep learning project to detect whether a person is wearing a face mask or not. It uses Transfer Learning with MobileNetV2 for efficient real-time inference and includes a Streamlit web interface.

## Project Structure

- `dataset/`: Place your training data here in `with_mask` and `without_mask` subdirectories.
- `models/`: Saved trained models will be stored here.
- `notebooks/`: Includes Jupyter notebook for initial EDA.
- `src/`: Python source code for data preprocessing, model building, training, evaluation, and prediction.
- `app/`: Streamlit web app for deployment.
- `utils/`: Helper scripts, including a mock data generator for testing the pipeline out-of-the-box.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).
2. Extract the dataset so that `project/dataset/` contains two folders: `with_mask` and `without_mask`.

**Alternatively**, to quickly test the code structure without downloading real data, you can generate dummy train/val images by running:
```bash
python utils/helper.py
```

## Training

To train the model, run from the root of the `project` directory:

```bash
python src/train.py
```
This will:
- Preprocess data (resize to 224x224, augment, split to train/val).
- Build MobileNetV2 transfer learning model.
- Train the model and save the best version to `models/best_model.h5`.

## Evaluation

To evaluate the saved model and see the Precision, Recall, F1-Score, and Confusion Matrix:

```bash
python src/evaluate.py
```

## Running the Web App

Launch the Streamlit app to test with your webcam or uploaded photos:

```bash
streamlit run app/app.py
```
