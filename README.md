# ELMo From Scratch

This repository has the code for ELMo in two variants:
- ELMo with `Character CNNs`
- ELMo with `Pretrained Embeddings (GloVe)`

The file `WordEmb.py` contains the code for the Pretrained Word Embeddings

The file `WordEmb.py` goes through all the steps:
- Preprocessing the data
    - Loading the data
    - Preprocessing the data
    - Creating the vocabulary
    - Creating the Dataset
- Pretraining the Embeddings
    - Creating the Model
    - Training the Model
    - Evaluating the Model
    - Saving the Model
- Downstream Task
    - Loading the Pretrained Model
    - Creating the Downstream Model
    - Training the Downstream Model
    - Evaluating the Downstream Model

The file `CharCNN.py` contains the code for the Character CNNs

The file `CharCNN.py` goes through all the steps:
- Preprocessing the data
    - Loading the data
    - Preprocessing the data
    - Creating the char vocabulary
    - Creating the Dataset
- Pretraining the Embeddings
    - Creating the CharCNN Model
    - Creating the ELMo Model
    - Training the Model
    - Evaluating the Model
    - Saving the Model
- Downstream Task
    - Loading the Pretrained Model
    - Creating the Downstream Model
    - Training the Downstream Model
    - Evaluating the Downstream Model

There is a parameter `DIR` in both the files which is the path to the directory where you want to save the best models.

The `train.csv` and the `test.csv` have to be present within a directory `data` in the same directory as the code.

### Main Libraries used in the code:

    PyTorch       -> create the models
    Pandas        -> load the data
    Gensim        -> load the pretrained embeddings
    NLTK          -> tokenize the data
    Scikit-Learn  -> evaluate the models
    WandB         -> log the metrics
    NumPy         -> for numerical operations
    tqdm          -> for progress bars
    torchinfo     -> for model summary