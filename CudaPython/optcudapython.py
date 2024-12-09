# code in Python
import numpy as np
from numba import cuda, types as numba_types
import math
import csv
import time

from numba import cuda, float32
import math

@cuda.jit
def logistic_regression_update_shared(X, y, weights, bias, lr, epochs, num_features, num_samples):
    # Declare shared memory dynamically
    shared_mem = cuda.shared.array(0, numba_types.float32)

    # Map dynamic shared memory
    shared_weights = shared_mem[:num_features]
    shared_bias = shared_mem[num_features:num_features + 1]

    # Calculate global thread ID
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Load weights and bias into shared memory
    if cuda.threadIdx.x < num_features:
        shared_weights[cuda.threadIdx.x] = weights[cuda.threadIdx.x]
    if cuda.threadIdx.x == 0:
        shared_bias[0] = bias[0]
    cuda.syncthreads()  # Synchronize after loading weights and bias into shared memory

    # Check if thread is within the number of samples
    if thread_id < num_samples:
        for epoch in range(epochs):
            z = shared_bias[0]
            for j in range(num_features):
                z += X[thread_id, j] * shared_weights[j]

            # Compute prediction using sigmoid function
            prediction = 1.0 / (1.0 + math.exp(-z))
            error = y[thread_id] - prediction

            # Update weights using atomic addition (strided within block)
            for j in range(cuda.threadIdx.x, num_features, cuda.blockDim.x):
                cuda.atomic.add(weights, j, lr * error * X[thread_id, j])

            # Synchronize threads after updating weights
            cuda.syncthreads()

            # Update bias using atomic addition
            if cuda.threadIdx.x == 0:
                cuda.atomic.add(bias, 0, lr * error)

            # Synchronize threads before next epoch
            cuda.syncthreads()

def initialize_weights(vocab_size):
    return np.random.uniform(-0.01, 0.01, vocab_size).astype(np.float32)

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

# Build Vocabulary
def build_vocab(tokenized_texts):
    return list(set(token for tokens in tokenized_texts for token in tokens))

def read_csv(filename):
    texts, labels = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    return texts, np.array(labels, dtype=np.float32)

def preprocess_texts(texts, stopwords):
    tokenized_texts = []
    for text in texts:
        tokens = tokenize(text)  # Tokenization
        tokens = remove_stopwords(tokens, stopwords)  # Remove stopwords
        tokenized_texts.append(tokens)
    return tokenized_texts

def predict(features, weights, bias):
    predictions = []
    for x in features:
        z = bias[0] + np.dot(x, weights)
        z = max(min(z, 10), -10)
        prediction = 1.0 / (1.0 + np.exp(-z))
        predictions.append(1 if prediction >= 0.5 else 0)
    return predictions

def main():
    start_time = time.time()  # Start the timer
    texts, labels = read_csv("nlp.csv")

    # Define your stopwords list
    stopwords = set([
        'a', 'the', 'in', 'on', 'at', 'to', 'and', 'for', 'is', 'this', 'of', 'it', 'you', 'with', 'as', 'are'
    ])

    # Tokenize and remove stopwords, then build vocabulary
    tokenized_texts = preprocess_texts(texts, stopwords=stopwords)
    vocab = build_vocab(tokenized_texts)  # Build the vocabulary
    vocab_size = len(vocab)
    num_features = vocab_size  # Assign vocab_size to num_features

    if vocab_size == 0:
        print("Error: Vocabulary is empty. Check your preprocessing steps.")
        return

    #print(f"Vocabulary: {vocab[:10]}")  # Print first 10 words of the vocabulary for inspection

    bow_matrix = np.zeros((len(tokenized_texts), vocab_size), dtype=np.float32)
    for i, tokens in enumerate(tokenized_texts):
        bow_matrix[i] = text_to_bow(tokens, vocab)  # Convert each tokenized text into Bag of Words
    bow_matrix /= (np.linalg.norm(bow_matrix, axis=1, keepdims=True) + 1e-8)
    bow_matrix *= len(vocab)

    d_X = cuda.to_device(bow_matrix)
    d_y = cuda.to_device(labels)

    weights = np.zeros(vocab_size, dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    d_weights = cuda.to_device(weights)
    d_bias = cuda.to_device(bias)

    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK  # Maximum threads per block for your GPU
    num_multiprocessors = device.MULTIPROCESSOR_COUNT     # Number of SMs (streaming multiprocessors)

    epochs = 1000

    # Optimal configuration
    threads_per_block = 32
    blocks_per_grid = 4*(math.ceil(num_features / threads_per_block))




    print(f"Blocks per grid: {blocks_per_grid}, Threads per block: {threads_per_block}")

    learning_rate = 0.01
    epochs = 1000

    # Calculate shared memory size
    shared_mem_size = (num_features + 1) * 4  # 4 bytes per float32

    print("Training on GPU...")
    # Launch the kernel for all epochs in a single execution
    logistic_regression_update_shared[blocks_per_grid, threads_per_block, 0, shared_mem_size](
        d_X, d_y, d_weights, d_bias, learning_rate, epochs, vocab_size, len(texts)
    )
    cuda.synchronize()  # Ensure GPU kernel execution is complete

    # Copy weights and bias back to host for inspection
    final_weights = d_weights.copy_to_host()
    final_bias = d_bias.copy_to_host()

    #print("Training completed.")
    #print(f"Final Bias: {final_bias[0]:.6f}")
    #print(f"First 4 Weights: {final_weights[:4]}")

    test_texts = [
        "I love it, so good and amazing!",
        "I am very satisfied with my purchase.",
        "This is exactly what I needed.",
        "This app crashes constantly and is unusable.",
        "This makes my daily tasks so much easier.",
        "Excellent service, I will buy again!",
        "It's terrible and awful",
        "The package arrived late and was damaged"
    ]

    # Process test texts
    test_tokenized = preprocess_texts(test_texts, stopwords)
    test_bow = np.zeros((len(test_tokenized), vocab_size), dtype=np.float32)
    for i, tokens in enumerate(test_tokenized):
        test_bow[i] = text_to_bow(tokens, vocab)  # Convert test tokenized text to BoW

    predictions = predict(test_bow, final_weights, final_bias)

    for text, prediction in zip(test_texts, predictions):
        print(f"'{text}' -> {'Positive' if prediction == 1 else 'Negative'}")

    # Execution time
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
