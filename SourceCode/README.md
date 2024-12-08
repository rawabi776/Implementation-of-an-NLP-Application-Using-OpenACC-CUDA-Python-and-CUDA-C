## Source Code :

The code is a Natural Language Processing (NLP) pipeline implemented in C with a focus on logistic regression. Here's a brief overview:

## Key Components:
1- Data Preparation:
- CSV Reader: Reads text samples and their labels from a CSV file.
- Tokenization: Breaks text into lowercase words and removes punctuation.
- Stopword Removal: Filters out common, non-informative words like "the," "is," etc.
- Vocabulary Building: Creates a dynamic vocabulary of unique words from the dataset.

2- Feature Extraction:
- Converts processed text into a Bag-of-Words (BoW) representation, a numerical vector indicating word occurrences.

3- Logistic Regression:
- Training: Trains a logistic regression model using gradient descent to optimize weights and bias.
- Prediction: Applies the trained model to classify new texts as positive or negative.

4- Testing:
- Evaluates the trained model on a set of predefined test texts, generating predictions for sentiment classification.

5- Profiling and Performance Optimization:
- Uses NVTX (NVIDIA Tools Extension) for profiling phases such as data preparation, tokenization, training, and prediction.
Program Flow:
 - Initialization:
    - Allocates memory for data structures (texts, labels, vocabulary).
  - Data Processing:
      - Tokenizes texts, removes stopwords, and generates BoW features.
  - Training:
      - Performs iterative weight updates using logistic regression.
  - Prediction:
      - Classifies new test texts using the trained model.
  - Cleanup:
      - Frees allocated memory to avoid leaks.
- Highlights:
  - The program employs modular design with distinct functions for each task.
  - Dynamic memory allocation and efficient resizing ensure scalability.
  - Profiling with NVTX helps optimize computational bottlenecks.

