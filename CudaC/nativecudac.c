# code in CUDA C
%%writefile 01-cuda-code.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>

#define INITIAL_VOCAB_CAPACITY 100
#define INITIAL_BOW_CAPACITY 100

// Function prototype for prediction
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

// Initialize TokenizedText
void init_tokenized_text(TokenizedText *tokens) {
    tokens->words = (char **)malloc(INITIAL_BOW_CAPACITY * sizeof(char *));
    tokens->word_count = 0;
    tokens->capacity = INITIAL_BOW_CAPACITY;
}

// Free memory in TokenizedText
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

// Add word to vocabulary
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

// Tokenize text
TokenizedText tokenize(const char *text) {
    TokenizedText tokens;
    init_tokenized_text(&tokens);

    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " .,!?");

    while (token != NULL) {
        for (int i = 0; token[i]; i++) token[i] = tolower(token[i]); // Convert to lowercase
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

// Logistic regression training using unified memory
void gpu_train_logistic_regression(
    int **X, int *y, double *weights, double *bias,
    int num_samples, int num_features, double lr, int epochs
) {
    int *X_flat, *d_y;
    double *d_weights, *d_bias;

    // Allocate unified memory
    cudaCheckError(cudaMallocManaged(&X_flat, num_samples * num_features * sizeof(int)));
    cudaCheckError(cudaMallocManaged(&d_y, num_samples * sizeof(int)));
    cudaCheckError(cudaMallocManaged(&d_weights, num_features * sizeof(double)));
    cudaCheckError(cudaMallocManaged(&d_bias, sizeof(double)));

    // Flatten X
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            X_flat[i * num_features + j] = X[i][j];
        }
    }

    // Copy labels
    for (int i = 0; i < num_samples; i++) {
        d_y[i] = y[i];
    }

    // Initialize weights and bias
    memset(d_weights, 0, num_features * sizeof(double));
    *d_bias = 0.0;

    // Kernel launch settings
    int blockSize = 256;
    int gridSize = (num_samples + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < epochs; epoch++) {
        logistic_regression_kernel<<<gridSize, blockSize>>>(X_flat, d_y, d_weights, d_bias, num_samples, num_features, lr);
        cudaCheckError(cudaDeviceSynchronize());
    }

    // Copy results
    memcpy(weights, d_weights, num_features * sizeof(double));
    *bias = *d_bias;

    // Free unified memory
    cudaFree(X_flat);
    cudaFree(d_y);
    cudaFree(d_weights);
    cudaFree(d_bias);
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
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

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

    gpu_train_logistic_regression(X, labels, weights, &bias, num_texts, vocab.size, 0.01, 1000);

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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nTotal Execution Time: %f ms\n", elapsedTime);

    for (int i = 0; i < num_texts; i++) {
        free(X[i]);
        free(texts[i]);
    }
    free(X);
    free(texts);
    free(labels);
    free(weights);
    free_vocab(&vocab);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}