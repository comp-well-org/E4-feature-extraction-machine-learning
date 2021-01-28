# E4 feature engineering and machine learning

## Contents
### Tools
Based on package of [eda-explorer](https://eda-explorer.media.mit.edu) (Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. Automatic identification of artifacts in electrodermal activity data.In Engineering in Medicine and Biology Conference. 2015.)

* **extract_E4_features.py**
* **EDA_Peak_Detection_Script.py**
* **load_files.py**

### Example code
**Notes: the example codes assume for each participant data are collected from baseline and session**
* **extract_feature.ipynb**: extract ACC,TEMP,EDA,EDA peaks,IBI features of E4 data
* **e4_stats_tests.ipynb**: performing paired t-tests using [SciPy](https://www.scipy.org/)
* **svm_hyperparam.ipynb**: tuning hyperparameters of svm model using [Scikit-Learn](https://scikit-learn.org/stable/)

## Requirements
numpy==1.16.2

scipy==1.2.1

pandas==0.24.1

scikit-learn==0.20.3

matplotlib>=2.1.2

PyWavelets==1.0.2

[hrv-analysis](https://pypi.org/project/hrv-analysis/)

## Author
Yifei (Winnie) Li, Akane Sano, [Rice University Computational Wellbeing Group](https://compwell.rice.edu/)


