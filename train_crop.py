# Offline train_crop.py
import joblib, json
from pathlib import Path
from app import train_crop_pipeline
if __name__=='__main__':
    train_crop_pipeline(save_artifacts=True, quick=False)
