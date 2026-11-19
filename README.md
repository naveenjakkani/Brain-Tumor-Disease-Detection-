# Brain Tumor Detection System

A Python-based web application that detects brain tumors from MRI images using machine learning. The project uses handcrafted image features (histogram, texture, edge) and a Random Forest classifier to categorize MRI scans into four classes.

## Key Features

- Web interface built with Flask
- MRI image upload and prediction
- Prediction history tracking
- Probability distribution chart for model confidence
- Feature extraction using GLCM, histogram statistics, and edge density
- Random Forest model trained on a labeled brain tumor dataset

## Repository Structure

- `app.py` - Flask application controlling routes, file upload, and prediction display
- `train_model.py` - Model training script that extracts features and trains the classifier
- `utils/feature_extraction.py` - Image preprocessing and feature extraction pipeline
- `utils/predict.py` - Loads the trained model and scaler to predict new images
- `templates/` - HTML templates for the web interface
- `static/` - CSS and JavaScript files for styling and client-side behavior
- `DataSet_M/` - Expected dataset folders for training and testing images
- `models/` - Saved model and scaler files
- `requirements.txt` - Python dependencies

## Technologies Used

- Python 3
- Flask
- scikit-learn
- scikit-image
- OpenCV
- NumPy
- Matplotlib
- Joblib

## Dataset

The application is designed to use an MRI dataset with the following folder structure:

```
DataSet_M/
  Training/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/
  Testing/
    glioma_tumor/
    meningioma_tumor/
    no_tumor/
    pituitary_tumor/
```

Each class folder should contain MRI image files in JPG or PNG format.

## Model and Training

The model is trained using `train_model.py`:

1. Reads images from `DataSet_M/Training/`.
2. Preprocesses images with Gaussian blur, median filter, CLAHE, and normalization.
3. Extracts features:
   - histogram statistics (mean, std, skewness, kurtosis)
   - GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation)
   - edge density from Canny edge detection
4. Standardizes features with `StandardScaler`
5. Trains a `RandomForestClassifier`
6. Saves the trained model and scaler to `models/`

After training, `utils/predict.py` loads the saved model and scaler and uses the same feature extraction pipeline to classify new uploads.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/buchakegajanan/Brain-Tumor-Detection-System.git
cd "Brain Tumor Detection System"
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (if `models/brain_tumor_rf.pkl` does not exist or you want to retrain):
```bash
python train_model.py
```

2. Run the web application:
```bash
python app.py
```

3. Open the browser at:
```
http://127.0.0.1:5000
```

4. Login with:
- Email: `admin@hospital.com`
- Password: `admin123`

5. Upload an MRI image to get a prediction.

## Notes

- The project uses only classical machine learning, not deep learning.
- The current login is a simple mock authentication for demonstration purposes.
- For production use, secure authentication, upload validation, and dataset privacy measures are required.

## Recommended Improvements

- Add dataset validation and augmentation
- Add cross-validation and model performance reporting
- Use a database to store prediction history
- Remove `venv/` from version control and use `.gitignore`
- Add unit tests and CI/CD

## License

This repository does not include an explicit license file. Add a license if you plan to share widely.
