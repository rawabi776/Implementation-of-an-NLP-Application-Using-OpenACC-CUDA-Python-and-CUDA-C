# code in Python
%%writefile 02-python-code.py
import cProfile
import pstats
import numpy as np
from numba import cuda
import math
import csv
import time
from math import exp

# Select the GPU device
cuda.select_device(0)

# GPU Kernel for Logistic Regression Training
@cuda.jit
def train_logistic_regression_gpu(X, y, weights, bias, lr, epochs):
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    num_samples, num_features = X.shape

    if row < num_samples:
        for epoch in range(epochs):
            # Compute the weighted sum (z)
            z = bias[0]
            for j in range(num_features):
                z += weights[j] * X[row, j]

            # Sigmoid function
            prediction = 1 / (1 + exp(-z))  # Use math.exp for compatibility

            # Error calculation
            error = y[row] - prediction

            # Update weights
            for j in range(num_features):
                cuda.atomic.add(weights, j, lr * error * X[row, j])

            # Update bias
            cuda.atomic.add(bias, 0, lr * error)


# Prediction function
def predict(features, weights, bias):
    z = bias + np.dot(features, weights)
    return 1 if 1 / (1 + np.exp(-z)) >= 0.5 else 0

# Read Data from CSV
def read_csv(filename):
    texts, labels = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    return texts, labels

# Tokenization
def tokenize(text):
    return [word.lower() for word in text.split()]

# Remove Stopwords
def remove_stopwords(tokens, stopwords):
    return [word for word in tokens if word not in stopwords]

# Convert Text to Bag of Words
def text_to_bow(text_tokens, vocab):
    bow = np.zeros(len(vocab), dtype=np.int32)
    for token in text_tokens:
        if token in vocab:
            index = vocab.index(token)
            bow[index] += 1
    return bow

# Optimized Logistic Regression Training on GPU
def logistic_regression_gpu(X, y, num_features, lr, epochs):
    num_samples = X.shape[0]

    # Allocate memory on the GPU
    d_X = cuda.to_device(X)
    d_y = cuda.to_device(y)
    d_weights = cuda.to_device(np.zeros(num_features, dtype=np.float32))
    d_bias = cuda.to_device(np.array([0.0], dtype=np.float32))

    # Grid and block configuration
    threads_per_block = 256  # Optimized thread count
    blocks_per_grid = max(1, (num_samples + threads_per_block - 1) // threads_per_block)

    print(f"Launching kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block.")

    # Launch GPU Kernel
    train_logistic_regression_gpu[blocks_per_grid, threads_per_block](
        d_X, d_y, d_weights, d_bias, lr, epochs
    )

    # Ensure kernel execution is complete before proceeding
    cuda.synchronize()

    # Copy results back to host
    weights = d_weights.copy_to_host()
    bias = d_bias.copy_to_host()
    return weights, bias[0]

# Main Program
def main():
    start_time = time.time()

    # Read data (upload your dataset as 'nlp.csv' to Colab or your local environment)
    texts, labels = read_csv("nlp.csv")

    # Tokenization and Stopword Removal
    stop_words = {"i", "this", "is", "a", "the", "it", "not", "and", "but", "very", "of"}
    tokenized_texts = [remove_stopwords(tokenize(text), stop_words) for text in texts]

    # Build Vocabulary
    vocab = list(set(token for tokens in tokenized_texts for token in tokens))

    # Convert Texts to BoW
    X = np.array([text_to_bow(tokens, vocab) for tokens in tokenized_texts], dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # Monitor memory usage
    print(f"X size: {X.nbytes / (1024 ** 2):.2f} MB")
    print(f"y size: {y.nbytes / (1024 ** 2):.2f} MB")

    # Training Parameters
    learning_rate = 0.01
    epochs = 1000

    print("Training on GPU...")
    weights, bias = logistic_regression_gpu(X, y, len(vocab), learning_rate, epochs)

    # Prediction
    test_texts = ["It's terrible and awful.", "Amazing! I like it.", "This is exactly what I needed.", "The worst decision Iâ€™ve ever made."]
    print("\nPredictions:")
    for test_text in test_texts:
        test_tokens = remove_stopwords(tokenize(test_text), stop_words)
        test_bow = text_to_bow(test_tokens, vocab)
        prediction = predict(test_bow, weights, bias)
        print(f"'{test_text}' -> {'Positive' if prediction == 1 else 'Negative'}")

    # Execution Time
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

# Add cProfile for Profiling
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    main()
    profiler.disable()  # Stop profiling

    # Save and print profiling results
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(10)  # Print top 10 functions by cumulative time