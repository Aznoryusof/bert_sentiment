# Sentiment Analysis with BERT
This project uses a pre-trained BERT model to predict the sentiments of movie reviews.

The BERT model is pre-trained on a large corpus of English data via self-supervision, i.e. without any labelling. During training, it has two objectives: Masked Language Modelling and Next Sentence Predictions. The model learns representations of the English language, which could be used as features in the sentiment classification task.

The model used in this project is the uncased version, and is fine-tuned on only the training data that had been provided.

## Setup
1. Install pytorch version 1.7.0 by following the instructions here
	 ```
   https://pytorch.org/
   ```
2. Next, go to the project directory titled "bert_sentiment" and install the necessary libraries
   ```
   pip install -r requirements.txt
   ```
   
## Download model artifacts
Next download the necessary model artifacts including the Bert Model that has been finetuned by running the following script.
```
python src/download_artifacts.py
```

## Perform inference on test file
Finally, perform predictions on the test file by running the following script in the terminal in the project's main directory. The file containing the predictions for the test is saved as *_pred.csv in the data directory.
```
python src/test.py data/Test-format.csv
```
Note: 
- The above command takes the path of the test data as the first positional argument. In this case "data/Test-format.csv".
- By default, inference uses GPU. This can be changed by setting "use_gpu_test=False" in the config.py file.

## Validation results
Validation can be done by running the following script.
```
python src/validation.py data/Valid.csv
```
The results of the validation is as follows:
<br> <img src="validation.png" width="400"/> <br>

## Resources
The following resources were referenced to implement this project:
- https://www.youtube.com/watch?v=_eSGWNqKeeY&t=1453s
