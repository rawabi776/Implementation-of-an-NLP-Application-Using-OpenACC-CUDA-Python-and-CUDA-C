#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <time.h>
#include <nvToolsExt.h>

#define INITIAL_VOCAB_CAPACITY 100
#define INITIAL_BOW_CAPACITY 100

typedef struct {
    char **words;
    int word_count;
    int capacity;
} TokenizedText;

typedef struct {
    char **words;
    int size;
    int capacity;
} Vocabulary;

// Function prototypes
void read_csv(const char *filename, char ***texts, int **labels, int *num_texts);
TokenizedText tokenize(const char *text);
TokenizedText remove_stopwords(const TokenizedText *tokens);
void free_tokenized_text(TokenizedText *tokens);
void init_vocab(Vocabulary *vocab);
void add_to_vocab(Vocabulary *vocab, const char *word);
void text_to_bow(const TokenizedText *text, Vocabulary *vocab, int *bow);
void free_vocab(Vocabulary *vocab);
int predict(int *features, double *weights, double bias, int num_features);
void train_logistic_regression(int **X, int *y, double *weights, double *bias, int num_samples, int num_features, double lr, int epochs);

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

TokenizedText tokenize(const char *text) {
    TokenizedText tokens;
    tokens.words = (char **)malloc(INITIAL_BOW_CAPACITY * sizeof(char *));
    tokens.word_count = 0;
    tokens.capacity = INITIAL_BOW_CAPACITY;

    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " .,!?");

    while (token != NULL) {
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
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

TokenizedText remove_stopwords(const TokenizedText *tokens) {
    const char *stop_words[] = {"i", "this", "is", "a", "the", "it", "not", "and", "but", "very", "of"};
    int num_stop_words = sizeof(stop_words) / sizeof(stop_words[0]);

    TokenizedText filtered;
    filtered.words = (char **)malloc(tokens->capacity * sizeof(char *));
    filtered.word_count = 0;
    filtered.capacity = tokens->capacity;

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

void free_tokenized_text(TokenizedText *tokens) {
    for (int i = 0; i < tokens->word_count; i++) {
        free(tokens->words[i]);
    }
    free(tokens->words);
}

void init_vocab(Vocabulary *vocab) {
    vocab->words = (char **)malloc(INITIAL_VOCAB_CAPACITY * sizeof(char *));
    vocab->size = 0;
    vocab->capacity = INITIAL_VOCAB_CAPACITY;
}

void add_to_vocab(Vocabulary *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++) {
        if (strcmp(vocab->words[i], word) == 0) return;
    }
    if (vocab->size == vocab->capacity) {
        vocab->capacity *= 2;
        vocab->words = (char **)realloc(vocab->words, vocab->capacity * sizeof(char *));
    }
    vocab->words[vocab->size++] = strdup(word);
}

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

void free_vocab(Vocabulary *vocab) {
    for (int i = 0; i < vocab->size; i++) {
        free(vocab->words[i]);
    }
    free(vocab->words);
}

int predict(int *features, double *weights, double bias, int num_features) {
    double z = bias;
    for (int i = 0; i < num_features; i++) {
        z += features[i] * weights[i];
    }
    return 1 / (1 + exp(-z)) >= 0.5 ? 1 : 0;
}

void train_logistic_regression(int **X, int *y, double *weights, double *bias, int num_samples, int num_features, double lr, int epochs) {
    #pragma acc data copyin(X[0:num_samples][0:num_features], y[0:num_samples]) \
                     copyout(weights[0:num_features]) create(bias)
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double bias_update = 0.0;

            #pragma acc parallel loop gang reduction(+:bias_update) vector_length(128)
            for (int i = 0; i < num_samples; i++) {
                double z = *bias;

                #pragma acc loop vector reduction(+:z)
                for (int j = 0; j < num_features; j++) {
                    z += X[i][j] * weights[j];
                }

                double prediction = 1.0 / (1.0 + exp(-z));
                double error = y[i] - prediction;

                #pragma acc loop vector
                for (int j = 0; j < num_features; j++) {
                    weights[j] += lr * error * X[i][j];
                }

                bias_update += lr * error;
            }

            *bias += bias_update;
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
    int **X = (int **)malloc(num_texts * sizeof(int *));
    for (int i = 0; i < num_texts; i++) {
        X[i] = (int *)malloc(vocab.size * sizeof(int));
        text_to_bow(&processed_texts[i], &vocab, X[i]);
        free_tokenized_text(&processed_texts[i]);
    }
    nvtxRangePop();

    nvtxRangePushA("Logistic Regression Training");
    double *weights = (double *)calloc(vocab.size, sizeof(double));
    double bias = 0.0;

    #pragma acc data copyin(X[0:num_texts][0:vocab.size], labels[0:num_texts]) \
                     copyout(weights[0:vocab.size]) copy(bias)
    train_logistic_regression(X, labels, weights, &bias, num_texts, vocab.size, 0.01, 1000);
    nvtxRangePop();

    nvtxRangePushA("Prediction Phase");
    const char *test_texts[] = {
        "It's terrible and awful.", "Amazing! I like it.",
        "Unbelievable quality so good!", "I want my money back awful.",
        "The food tasted bland and uninspired.",
        "I couldn't be happier with the customer service!",
        "The battery life is atrocious and drains quickly.",
        "An exceptional product that delivers on all promises.",
        "The package arrived late and was damaged.",
        "I was blown away by the quality of this item!",
        "The instructions were confusing and poorly written.",
        "Such a delightful experience shopping with them."
    };
    for (int i = 0; i < 12; i++) {
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
