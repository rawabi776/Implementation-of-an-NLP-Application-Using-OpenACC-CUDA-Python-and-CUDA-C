##  Key Components for the Native OpenACC and the Optimized OpenACC Codes: 
1- Data Preparation:
- Both implementations process CSV input files for text and labels.They tokenize text, remove stopwords, and build a vocabulary.
- Bag-of-Words (BoW) representations are created for classification.

2- Logistic Regression:
- Both versions use logistic regression for text classification.
- Gradient descent optimizes weights and bias through iterative updates.
- #pragma acc directives handle parallelization.

3- Prediction:
- Implements sentiment analysis using the trained model.
- Outputs "Positive" or "Negative" sentiment for test samples.

4- Memory Management:
- Dynamic memory allocation for text data, vocabulary, and BoW arrays.
- Proper cleanup through deallocation functions.

5- Profiling with NVTX:

- Adds NVTX (NVIDIA Tools Extension) profiling markers to measure execution times for major phases:
    - Data Preparation
    - Tokenization and Stopword Removal
    - Logistic Regression Training
    - Prediction Phase

## Enhancements in Optimized OpenACC Code: 

1- Improved Parallelism:
- Vector Length Optimization: Adjusts vector length for finer control of parallel resources.
- Example: vector_length(128) in #pragma acc parallel loop gang reduction.
- Nested Parallel Loops: Uses vectorized operations in loops for bias and weight updates, improving efficiency.

2- Data Movement Optimization:
- Reduces redundant data transfers by better leveraging #pragma acc data clauses.
- Example: copyin and copyout operations ensure efficient memory synchronization between CPU and GPU.

3- Bias Handling:
- Incorporates bias_update as a reduced variable to minimize overhead during iterative updates.

4- Memory Allocation Enhancements:
- Efficient resizing strategies for arrays reduce overhead from frequent reallocations.
