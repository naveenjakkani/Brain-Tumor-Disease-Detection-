import os
import glob
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils.feature_extraction import extract_features

def main():
    base_dir = r"c:\Users\gajan\Desktop\PROJECT\Healthcare\Brain Tumor Detection System\DataSet_M\Training"
    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    
    print("Extracting features from Training dataset...")
    
    X = []
    y = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue
            
        # Get all jpg/png files
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))
        
        print(f"Processing {class_name}: found {len(image_paths)} images")
        
        for i, img_path in enumerate(image_paths):
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(class_idx)
                
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1} images...")
                
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total extracted features shape: {X.shape}")
    print(f"Total labels shape: {y.shape}")
    
    if len(X) == 0:
        print("No images found or extracted. Aborting training.")
        return
        
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training Random Forest Classifier...")
    # Initialize Random Forest (using 100 trees, random state for reproducibility)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_scaled, y)
    
    # Quick training evaluation
    y_pred = rf_model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    print(classification_report(y, y_pred, target_names=classes))
    
    # Save models
    models_dir = r"c:\Users\gajan\Desktop\PROJECT\Healthcare\Brain Tumor Detection System\models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "brain_tumor_rf.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Evaluate on test set
    test_base_dir = base_dir.replace("Training", "Testing")
    print("Evaluating on Test dataset...")
    
    X_test = []
    y_test = []
    
    for class_idx, class_name in enumerate(classes):
        test_class_dir = os.path.join(test_base_dir, class_name)
        if not os.path.exists(test_class_dir):
            print(f"Warning: Test directory {test_class_dir} does not exist. Skipping.")
            continue
            
        # Get all jpg/png files
        test_image_paths = glob.glob(os.path.join(test_class_dir, "*.jpg")) + glob.glob(os.path.join(test_class_dir, "*.png"))
        
        print(f"Processing test {class_name}: found {len(test_image_paths)} images")
        
        for img_path in test_image_paths:
            features = extract_features(img_path)
            if features is not None:
                X_test.append(features)
                y_test.append(class_idx)
                
    if len(X_test) > 0:
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = rf_model.predict(X_test_scaled)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=classes))
    else:
        print("No test images found. Skipping test evaluation.")
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()
