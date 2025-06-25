import kagglehub

# Download the dataset from Kaggle to local path
path = kagglehub.dataset_download(
    "computerscience3/public-requirementspure-dataset")

print("Path to dataset files:", path)

