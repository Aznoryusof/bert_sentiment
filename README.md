# Sentiment Analysis with BERT
This project uses a pre-trained BERT model, fine-tuned on the training data provided to predict sentiments of texts.

## Setup
Firstly, install the necessary libraries
```
pip install -r requirements.txt
```

## Download model artifacts
Next download the necessary model artifacts including the Bert Model that has been finetuned by running the following script in the terminal in the project's main directory.
```
python src/download_artifacts.py
```

## Perform inference on test file
Finally, perform predictions on the test file by running the following script in the terminal in the project's main directory.
```
python src/test.py data/Test-format.csv
```
Note: 
- The above command takes the path of the test data as the first positional argument.
- By default, inference uses GPU. This can be changed by setting "use_gpu_test=False" in the config.py file.

## Resources 
The following resources were referenced to implement this project:
- https://www.youtube.com/watch?v=_eSGWNqKeeY&t=1453s