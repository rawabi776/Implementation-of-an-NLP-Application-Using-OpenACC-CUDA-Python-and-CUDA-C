#include <stdio.h>   // For file I/O and printing
#include <stdlib.h>  // For memory management
#include <string.h>  // For string operations
#include <ctype.h>   // For character operations
#include <math.h>    // For math functions like exp
#include <stddef.h>  // For size_t
#include <time.h>    // For measuring execution time
#include <nvToolsExt.h> // For NVTX profiling

#define INITIAL_VOCAB_CAPACITY 100
#define INITIAL_BOW_CAPACITY 100

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
    tokens->words = malloc(INITIAL_BOW_CAPACITY * sizeof(char *));
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
    vocab->words = malloc(INITIAL_VOCAB_CAPACITY * sizeof(char *));
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
        vocab->words = realloc(vocab->words, vocab->capacity * sizeof(char *));
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
            tokens.words = realloc(tokens.words, tokens.capacity * sizeof(char *));
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
                filtered.words = realloc(filtered.words, filtered.capacity * sizeof(char *));
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

// Read CSV file
void read_csv(const char *filename, char ***texts, int **labels, int *num_texts) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    int capacity = INITIAL_BOW_CAPACITY;
    *texts = malloc(capacity * sizeof(char *));
    *labels = malloc(capacity * sizeof(int));

    char line[1024];
    *num_texts = 0;

    while (fgets(line, sizeof(line), file)) {
        if (*num_texts == capacity) {
            capacity *= 2;
            *texts = realloc(*texts, capacity * sizeof(char *));
            *labels = realloc(*labels, capacity * sizeof(int));
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

// Logistic regression prediction
int predict(int *features, double *weights, double bias, int num_features) {
    double z = bias;
    for (int i = 0; i < num_features; i++) {
        z += features[i] * weights[i];
    }
    return 1 / (1 + exp(-z)) >= 0.5 ? 1 : 0;
}

// Train logistic regression
void train_logistic_regression(int **X, int *y, double *weights, double *bias, int num_samples, int num_features, double lr, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            int prediction = predict(X[i], weights, *bias, num_features);
            double error = y[i] - prediction;

            for (int j = 0; j < num_features; j++) {
                weights[j] += lr * error * X[i][j];
            }
            *bias += lr * error;
        }
    }
}

int main() {
    clock_t start, end;
    start = clock();

    char **texts;
    int *labels;
    int num_texts;

    nvtxRangePushA("Data Preparation");
    read_csv("nlp.csv", &texts, &labels, &num_texts);
    nvtxRangePop();

    nvtxRangePushA("Tokenization and Stopword Removal");
    TokenizedText processed_texts[num_texts];
    for (int i = 0; i < num_texts; i++) {
        TokenizedText tokens = tokenize(texts[i]);
        processed_texts[i] = remove_stopwords(&tokens);
        free_tokenized_text(&tokens);
    }
    nvtxRangePop();

    nvtxRangePushA("Vocabulary Building");
    Vocabulary vocab;
    init_vocab(&vocab);
    for (int i = 0; i < num_texts; i++) {
        for (int j = 0; j < processed_texts[i].word_count; j++) {
            add_to_vocab(&vocab, processed_texts[i].words[j]);
        }
    }
    nvtxRangePop();

    nvtxRangePushA("Convert Texts to BoW");
    int **X = malloc(num_texts * sizeof(int *));
    for (int i = 0; i < num_texts; i++) {
        X[i] = malloc(vocab.size * sizeof(int));
        text_to_bow(&processed_texts[i], &vocab, X[i]);
        free_tokenized_text(&processed_texts[i]);
    }
    nvtxRangePop();

    nvtxRangePushA("Logistic Regression Training");
    double *weights = calloc(vocab.size, sizeof(double));
    double bias = 0.0;
    train_logistic_regression(X, labels, weights, &bias, num_texts, vocab.size, 0.01, 1000);
    nvtxRangePop();

    nvtxRangePushA("Prediction Phase");
    const char *test_texts[] = {"It's terrible and awful.", "Amazing! I like it.","Unbelievable quality so good!","I want my money back awful.", "The food tasted bland and uninspired.",
" I couldn't be happier with the customer service! ",
 "The battery life is atrocious and drains quickly. ",
" An exceptional product that delivers on all promises.",
 "The package arrived late and was damaged. ",
" I was blown away by the quality of this item! ",
 "The instructions were confusing and poorly written. ",
 "Such a delightful experience shopping with them." };
    for (int i = 0; i < 12; i++) {
        TokenizedText tokens = tokenize(test_texts[i]);
        TokenizedText cleaned_tokens = remove_stopwords(&tokens);
        int *test_bow = calloc(vocab.size, sizeof(int));
        text_to_bow(&cleaned_tokens, &vocab, test_bow);

        int prediction = predict(test_bow, weights, bias, vocab.size);
        printf("'%s' -> %s\n", test_texts[i], prediction == 1 ? "Positive" : "Negative");

        free(test_bow);
        free_tokenized_text(&tokens);
        free_tokenized_text(&cleaned_tokens);
    }
    nvtxRangePop();

    for (int i = 0; i < num_texts; i++) {
        free(X[i]);
        free(texts[i]);
    }
    free(X);
    free(texts);
    free(labels);
    free(weights);
    free_vocab(&vocab);

    end = clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Total Execution Time: %f seconds\n", total_time);

    return 0;
}
