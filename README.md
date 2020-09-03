# ML_tools
Collection of general ML tools for a variety of datasets

# Installation
```
git clone https://github.com/jtjanecek/ML_tools
cd ML_tools
pip install -r requirements.txt
```

# Usage
general_classifier.py:
- Runs LogisticRegression, SVM, and RandomForest classifiers on an input CSV dataset
- CSV input must have a column labeled 'outcome'
- Feature labels should be above feature values

### Running
```
python general_classifier.py --input INPUT --output OUTPUT_DIR --cv CV
```
Where:
- INTPUT = CSV dataset
- OUTPUT = output directory (default = current directory)
- CV = python cross validation metric (default = LeaveOneOut), other options KFold etc
