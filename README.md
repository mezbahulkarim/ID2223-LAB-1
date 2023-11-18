# ID2223-LAB-1

Serverless Machine Learning model trained on the Iris and Wine Quality Datasets

Source Code for /iris: https://github.com/ID2223KTH/id2223kth.github.io/tree/master/src/serverless-ml-intro

## To run the project run the files in this suggested order

1. *-eda-and-backfill-feature-group.ipynb
2. *-training-pipeline.ipynb
3. *-feature-pipeline-daily.py
4. *-batch-inference-pipeline.py

## Background on some specific Python Packages 

### Hopsworks 

This is used as the "feature store" where all the features of the dataset are stored. This is done in Step 1 above. 
Furthermore, this is also used to store the trained machine learning model in Step 2 above. 

### Modal

Runs containerized code in the cloud. In the project it is used to add features to our "feaure store" in Step 3 above.
In Step 4 the trained model in Step 2 is used to make a prediction by running a container on modal.  

### Gradio

This is used to build the UI for the project in both /huggingface* directories.
Run the app.py files to make a prediction on the data or to monitor the performance of the model.


