# code in CUDA C
%%writefile 03-cuda-code.cu
#include <stdio.h>   // For file I/O and printing
#include <stdlib.h>  // For memory management
#include <string.h>  // For string operations
#include <ctype.h>   // For character operations
#include <math.h>    // For math functions like exp
#include <cuda_runtime.h> // For CUDA functions

#define INITIAL_VOCAB_CAPACITY 100
#define INITIAL_BOW_CAPACITY 100

// Function prototype for predict
int predict(int *features, double *weights, double bias, int num_features);
int predict(int *features, double *weights, double bias, int num_features) {
    double z = bias;
    for (int i = 0; i < num_features; i++) {
        z += features[i] * weights[i];
    }
    return 1 / (1 + exp(-z)) >= 0.5 ? 1 : 0;
}

// Macro for CUDA error handling
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Structure for a tokenized text
typedef struct {
    char **words;
    int word_count;
    int capacity;
} TokenizedText;

// Structure for dynamic vocabulary
typedef struct {
    char **words;
    int size;
    int capacity;
} Vocabulary;

// Function to initialize TokenizedText
void init_tokenized_text(TokenizedText *tokens) {
    tokens->words = (char **)malloc(INITIAL_BOW_CAPACITY * sizeof(char *));
    tokens->word_count = 0;
    tokens->capacity = INITIAL_BOW_CAPACITY;
}

// Free TokenizedText memory
void free_tokenized_text(TokenizedText *tokens) {
    for (int i = 0; i < tokens->word_count; i++) {
        free(tokens->words[i]);
    }
    free(tokens->words);
}

// Initialize dynamic vocabulary
void init_vocab(Vocabulary *vocab) {
    vocab->words = (char **)malloc(INITIAL_VOCAB_CAPACITY * sizeof(char *));
    vocab->size = 0;
    vocab->capacity = INITIAL_VOCAB_CAPACITY;
}

// Free vocabulary memory
void free_vocab(Vocabulary *vocab) {
    for (int i = 0; i < vocab->size; i++) {
        free(vocab->words[i]);
    }
    free(vocab->words);
}

// Add word to vocabulary if not already present
void add_to_vocab(Vocabulary *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++) {
        if (strcmp(vocab->words[i], word) == 0) return; // Already in vocab
    }
    if (vocab->size == vocab->capacity) {
        vocab->capacity *= 2;
        vocab->words = (char **)realloc(vocab->words, vocab->capacity * sizeof(char *));
    }
    vocab->words[vocab->size++] = strdup(word);
}

// Tokenize the text
TokenizedText tokenize(const char *text) {
    TokenizedText tokens;
    init_tokenized_text(&tokens);

    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " .,!?");

    while (token != NULL) {
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]); // Convert to lowercase
        }
        if (tokens.word_count == tokens.capacity) {
            tokens.capacity *= 2;
            tokens.words = (char **)realloc(tokens.words, tokens.capacity * sizeof(char *));
        }
        tokens.words[tokens.word_count++] = strdup(token);
        token = strtok(NULL, " .,!?");
    }
    free(text_copy);
    return tokens;
}

// Remove stopwords
TokenizedText remove_stopwords(const TokenizedText *tokens) {
    const char *stop_words[] = {"i", "this", "is", "a", "the", "it", "not", "and", "but", "very", "of"};
    int num_stop_words = sizeof(stop_words) / sizeof(stop_words[0]);

    TokenizedText filtered;
    init_tokenized_text(&filtered);

    for (int i = 0; i < tokens->word_count; i++) {
        int is_stopword = 0;
        for (int j = 0; j < num_stop_words; j++) {
            if (strcmp(tokens->words[i], stop_words[j]) == 0) {
                is_stopword = 1;
                break;
            }
        }
        if (!is_stopword) {
            if (filtered.word_count == filtered.capacity) {
                filtered.capacity *= 2;
                filtered.words = (char **)realloc(filtered.words, filtered.capacity * sizeof(char *));
            }
            filtered.words[filtered.word_count++] = strdup(tokens->words[i]);
        }
    }
    return filtered;
}

// Convert text to Bag of Words
void text_to_bow(const TokenizedText *text, Vocabulary *vocab, int *bow) {
    memset(bow, 0, vocab->size * sizeof(int));
    for (int i = 0; i < text->word_count; i++) {
        for (int j = 0; j < vocab->size; j++) {
            if (strcmp(text->words[i], vocab->words[j]) == 0) {
                bow[j]++;
                break;
            }
        }
    }
}

// Logistic regression GPU kernel
__global__ void logistic_regression_kernel(
    int *X, int *y, double *weights, double *bias,
    int num_samples, int num_features, double lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        double z = *bias;
        for (int j = 0; j < num_features; j++) {
            z += X[idx * num_features + j] * weights[j];
        }
        double prediction = 1 / (1 + exp(-z));
        double error = y[idx] - prediction;

        for (int j = 0; j < num_features; j++) {
            atomicAdd(&weights[j], lr * error * X[idx * num_features + j]);
        }
        atomicAdd(bias, lr * error);
    }
}

// Function to train logistic regression using CUDA streams
void gpu_train_logistic_regression_with_streams(
    int **X, int *y, double *weights, double *bias,
    int num_samples, int num_features, double lr, int epochs
) {
    int *d_X, *d_y;
    double *d_weights, *d_bias;

    // Flatten the 2D array X into a 1D array for CUDA memory
    int *X_flat = (int *)malloc(num_samples * num_features * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            X_flat[i * num_features + j] = X[i][j];
        }
    }

    // Allocate memory on the GPU
    cudaCheckError(cudaMalloc((void **)&d_X, num_samples * num_features * sizeof(int)));
    cudaCheckError(cudaMalloc((void **)&d_y, num_samples * sizeof(int)));
    cudaCheckError(cudaMalloc((void **)&d_weights, num_features * sizeof(double)));
    cudaCheckError(cudaMalloc((void **)&d_bias, sizeof(double)));

    // Initialize GPU weights and bias
    cudaCheckError(cudaMemcpy(d_weights, weights, num_features * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_bias, bias, sizeof(double), cudaMemcpyHostToDevice));

    // Get GPU properties to determine the number of streams dynamically
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0); // Assume device 0 for simplicity
      int num_SMs = deviceProp.multiProcessorCount;

    // Dynamic stream allocation
    int num_streams = deviceProp.multiProcessorCount; // You can adjust this based on your GPU capacity
    size_t chunk_size = (num_samples + num_streams - 1) / num_streams;
    cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Asynchronous memory transfers using streams
    for (int i = 0; i < num_streams; i++) {
        size_t offset = i * chunk_size * num_features;
        size_t sample_offset = i * chunk_size;
        size_t current_chunk_size = (i == num_streams - 1) ? (num_samples - sample_offset) : chunk_size;

        cudaCheckError(cudaMemcpyAsync(d_X + offset, X_flat + offset, current_chunk_size * num_features * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
        cudaCheckError(cudaMemcpyAsync(d_y + sample_offset, y + sample_offset, current_chunk_size * sizeof(int), cudaMemcpyHostToDevice, streams[i]));
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Kernel launch settings
        int blockSize = 256;
        for (int i = 0; i < num_streams; i++) {
            size_t sample_offset = i * chunk_size;
            size_t current_chunk_size = (i == num_streams - 1) ? (num_samples - sample_offset) : chunk_size;
            int gridSize = 32*num_SMs;

            logistic_regression_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                d_X + sample_offset * num_features, d_y + sample_offset, d_weights, d_bias, current_chunk_size, num_features, lr
            );
        }

        // Synchronize all streams after each epoch
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // Copy final weights and bias back to host
    cudaCheckError(cudaMemcpy(weights, d_weights, num_features * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(bias, d_bias, sizeof(double), cudaMemcpyDeviceToHost));

    // Clean up GPU memory and streams
    cudaCheckError(cudaFree(d_X));
    cudaCheckError(cudaFree(d_y));
    cudaCheckError(cudaFree(d_weights));
    cudaCheckError(cudaFree(d_bias));
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    free(streams);
    free(X_flat);
}


// Read CSV file
void read_csv(const char *filename, char ***texts, int **labels, int *num_texts) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    int capacity = INITIAL_BOW_CAPACITY;
    *texts = (char **)malloc(capacity * sizeof(char *));
    *labels = (int *)malloc(capacity * sizeof(int));

    char line[1024];
    *num_texts = 0;

    while (fgets(line, sizeof(line), file)) {
        if (*num_texts == capacity) {
            capacity *= 2;
            *texts = (char **)realloc(*texts, capacity * sizeof(char *));
            *labels = (int *)realloc(*labels, capacity * sizeof(int));
        }
        char *text = strtok(line, ",");
        char *label = strtok(NULL, ",");
        if (text && label) {
            (*texts)[*num_texts] = strdup(text);
            (*labels)[*num_texts] = atoi(label);
            (*num_texts)++;
        }
    }

    fclose(file);
}

// Main function
int main() {
    cudaEvent_t start_total, stop_total;
    float total_time;

    cudaCheckError(cudaEventCreate(&start_total));
    cudaCheckError(cudaEventCreate(&stop_total));
    cudaCheckError(cudaEventRecord(start_total, 0));

    char **texts;
    int *labels;
    int num_texts;

    read_csv("nlp.csv", &texts, &labels, &num_texts);
    printf("Loaded %d samples from dataset.\n", num_texts);

    Vocabulary vocab;
    init_vocab(&vocab);

    TokenizedText processed_texts[num_texts];
    for (int i = 0; i < num_texts; i++) {
        TokenizedText tokens = tokenize(texts[i]);
        processed_texts[i] = remove_stopwords(&tokens);
        free_tokenized_text(&tokens);
    }

    for (int i = 0; i < num_texts; i++) {
        for (int j = 0; j < processed_texts[i].word_count; j++) {
            add_to_vocab(&vocab, processed_texts[i].words[j]);
        }
    }

    int **X = (int **)malloc(num_texts * sizeof(int *));
    for (int i = 0; i < num_texts; i++) {
        X[i] = (int *)malloc(vocab.size * sizeof(int));
        text_to_bow(&processed_texts[i], &vocab, X[i]);
        free_tokenized_text(&processed_texts[i]);
    }

    double *weights = (double *)calloc(vocab.size, sizeof(double));
    double bias = 0.0;

    gpu_train_logistic_regression_with_streams(X, labels, weights, &bias, num_texts, vocab.size, 0.01, 1000);

    printf("\nPrediction Phase:\n");
    const char *test_texts[] = {
        "This is amazing and wonderful!",
        "I wasted my money on this useless item.",
        "This is exactly what I needed.",
        "It is terrible and bad."
    };
    int num_test_texts = sizeof(test_texts) / sizeof(test_texts[0]);

    for (int i = 0; i < num_test_texts; i++) {
        TokenizedText tokens = tokenize(test_texts[i]);
        TokenizedText cleaned_tokens = remove_stopwords(&tokens);

        int *test_bow = (int *)calloc(vocab.size, sizeof(int));
        text_to_bow(&cleaned_tokens, &vocab, test_bow);

        int prediction = predict(test_bow, weights, bias, vocab.size);
        printf("'%s' -> %s\n", test_texts[i], prediction == 1 ? "Positive" : "Negative");

        free(test_bow);
        free_tokenized_text(&tokens);
        free_tokenized_text(&cleaned_tokens);
    }

    cudaCheckError(cudaEventRecord(stop_total, 0));
    cudaCheckError(cudaEventSynchronize(stop_total));
    cudaCheckError(cudaEventElapsedTime(&total_time, start_total, stop_total));
    printf("\nTotal Execution Time: %f ms\n", total_time);

    for (int i = 0; i < num_texts; i++) {
        free(X[i]);
        free(texts[i]);
    }
    free(X);
    free(texts);
    free(labels);
    free(weights);
    free_vocab(&vocab);

    cudaCheckError(cudaEventDestroy(start_total));
    cudaCheckError(cudaEventDestroy(stop_total));

    return 0;
}