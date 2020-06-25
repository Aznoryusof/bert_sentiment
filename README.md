# Sentiment Analysis with BERT
A project on fine tuning BERT with hotel reviews for sentiment analysis.

## Installation
Firstly, clone the repository to a folder of your choice. 

Next install the necessary libraries

1. Create an environment with a specific version of Python and activate the environment
	```
   conda create -n <env name> python=3.8
   ```

2. Install the appropriate version of pytorch by following the instructions here
	```
   https://pytorch.org/
   ```

3. Install libraries in requirements.txt file
   ```
   pip install -r requirements.txt
   ```

Next download the necessary datasets that have been pre-processed by running the following script in the terminal
in the project's main directory.

```
python src\make_dataset.py
```

Then process the original dataset to obtain a dataset suitable for BERT fine-tuning by running the following script in the terminal
in the project's main directory.

```
python src\process_data.py
```

Finally run the following file (similar to the above steps), to carry out model fine-tuning.

```
python src\train.py
```

To make predictions on text, change the prediction_text variable in the config.py file, and run the following file

```
python src\evaluate\predict.py
```

## Project Objective
Use Machine Learning and Natural Language Processing techniques to learn about the semantics and meanings of texts to distinguish hotel reviews that are negative or positive. This allows users to analyse sentiments of their customers at scale, and in real-time without much effort.

## Data Source
[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/kernels)

## Resources 
The following resources were referenced to implement this project:
- https://www.youtube.com/watch?v=_eSGWNqKeeY&t=1453s

# Further Development Work
- Building an app with Django
- Integrating app with portfolio website 
