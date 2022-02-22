# Named Entity Recognition (NER) with Deep Learning Model

## Abstract
This project is the homework 4 for USC CSCI-544 Applied Natural Language Proccessing.
This assignment gives you hands-on experience on building deep learning
models on named entity recognition (NER). We will use the CoNLL-2003
corpus to build a neural network for NER. The same as HW3, in the folder
named data, there are three files: train, dev and test. In the files of train and
dev, we provide you with the sentences with human-annotated NER tags.
In the file of test, we provide only the raw sentences. The data format is
that, each line contains three items separated by a white space symbol. The
first item is the index of the word in the sentence. The second item is the
word type and the third item is the corresponding NER tag. There will be a
blank line at the end of one sentence. We also provide you with a file named
glove.6B.100d.gz, which is the GloVe word embeddings.
We also provide the official evaluation script conll03eval to evaluate the
results of the model. To use the script, you need to install perl and prepare
your prediction file in the following format:

idx word gold pred

where there is a white space between two columns. gold is the gold-standard
NER tag and pred is the model-predicted tag. Then execute the command
line:

```bash
perl conll03eval < {predicted f ile}
```

where {predicted f ile} is the prediction file in the prepared format.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following packages or with Anaconda:

- pandas
- numpy
- PyTorch
- perl
- sklearn


The packages can be installed by following command:
```bash
pip install <package>
```

## Data
In the folder named 'data', there are three files: train, dev and test.
Please put train, dev and test files in the same directory with 'success_training.ipynb', 'success_testing.ipynb','success_testing.py' before execution.

## Execution
#### 1. model generation
For the 'success_training.ipynb' file, check following before execute it.
1. Put train, dev and test files in the same directory with 'success_training.ipynb' before execution.
2. Change the 'path' variable in the last line of cell 2 to current path.
3. Delete following lines in cell 2 if the file is not executed in Google Colab.

```bash
  #from google.colab import drive
  #drive.mount('/content/drive')
  #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Please run 'success_training.ipynb' file to generate following outputs.
- embedding_weight.pt: The initial embedding weight file for the trained model in Task 1.
- blstm1.pt: A model file for the trained model in Task 1.
- blstm2.pt: A model file for the trained model in Task 2.
These 3 files can be found in 'intermediate output' folder.

#### 2. check accuracy
After get 'blstm1.pt', 'blstm2.pt', and 'embedding_weight.pt' files, please	put the following files into same directory
- embedding_weight.pt: The initial embedding weight file for the trained model in Task 1.
- blstm1.pt: A model file for the trained model in Task 1.
- blstm2.pt: A model file for the trained model in Task 2.
- train
- test
- dev
- glove.6B.100d.txt
- success_testing.ipynb or success_testing.py (depends on you want to use Python or Jupyter Notebook)
For the 'glove.6B.100d.txt' file, please refer to the following Google Drive link and unzip it. <br>
[Google Drive link](https://drive.google.com/drive/folders/16mjkiI-SpqejIu6eb3o2rkW2M9EPTT-o?usp=sharing)


If you want to use python version:

In command line, go to this directory and run following command:
```bash
python3 success_testing.py <PATH_of current directory>
```

If you want to use Jupyter Notebook version:

uncomment line 20 and line 21 and run success_testing.ipynb file.

After executing python code, it will produce following output files:
- dev1.out    
- dev2.out    
- dev1_with_gold.out    
- dev2_with_gold.out    
- test1.out    
- test2.out

These above files can be found in 'accuracy output' folder.
dev1.out, dev2.out, test1.out and test2.out are the predictions of both dev and test data from Task 1 and Task 2, respectively.
All these files should be in the same format of training data.


## Step
### Task 1: Simple Bidirectional LSTM model
The first task is to build a simple bidirectional LSTM model for NER.

#### Task. Implementing the bidirectional LSTM network with PyTorch. The
architecture of the network is:

```bash
Embedding → BLSTM → Linear → ELU → classifier
```

The hyper-parameters of the network are listed in the following table:

```bash
embedding dim: 100
number of LSTM layers: 1
LSTM hidden dim: 256
LSTM Dropout: 0.33
Linear output dim: 128
```

Train this simple BLSTM model with the training data on NER with SGD
as the optimizer. Please tune other parameters that are not specified in the
above table, such as batch size, learning rate and learning rate scheduling.
What are the precision, recall and F1 score on the dev data?
(the reasonable F1 score on dev is about 77%.)

#### Task 2: Using GloVe word embeddings
The second task is to use the GloVe word embeddings to improve the BLSTM
in Task 1. The way we use the GloVe word embeddings is straight forward:
we initialize the embeddings in our neural network with the corresponding
vectors in GloVe. Note that GloVe is case-insensitive, but our NER model
should be case-sensitive because capitalization is an important information
for NER. You are asked to find a way to deal with this conflict. What are
the precision, recall and F1 score on the dev data?
(the reasonable F1
score on dev is about 88%.)
