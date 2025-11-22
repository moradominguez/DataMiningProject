# Ultra-Processed Food Binary Classifier

##  Project Overview
This project builds binary classifiers to identify **non–ultra-processed foods (class 1)** versus **ultra-processed foods (class 0)** from nutritional and categorical data.

### Positive Class (1):
Non–ultra-processed foods → labels `[0, 1, 2]`

### Negative Class (0):
Ultra-processed foods → label `[3]`

---
##  Folder Structure
See `src/` for preprocessing, training, and evaluation scripts.

##  How to Run
Create environment 
python -m venv venv

install requirements
pip install -r requirements.txt

run pipeline

```bash
python -m src.run_pipeline product_data.csv
