# Implementation-of-an-NLP-Application-Using-OpenACC-CUDA-Python-and-CUDA-C

## Team Members:
- Rawabi AlQahtani
- Najybah Al Talib
- Jawad Al Marhoon

## Instructor: 
- Dr. Ayaz ul Hassan Khan

## Description 
This project focuses on optimizing a Natural Language Processing (NLP) pipeline by parallelizing computationally intensive tasks using GPU programming techniques. The application was implemented using three models: OpenACC, CUDA Python, and CUDA C, and evaluated for their effectiveness in enhancing performance.


## Features

1- Data Preparation: Tokenization, stopword removal, and bag-of-words (BoW) feature extraction.
2- Logistic Regression Training: Optimized matrix computations using GPU parallelization.
3- Prediction: Efficient inference for new text samples.
4- Profiling and Analysis: Comprehensive performance evaluation using different GPU models.


## Technologies Used

1- OpenACC: High-level directives for GPU acceleration.
2- CUDA Python: Leveraging Python's flexibility and GPU acceleration libraries (e.g., Numba).
3- CUDA C: Fine-grained control over GPU resources for maximum performance.

## Methodology

1- Profiling Identified bottlenecks in the CPU-based implementation:
- Logistic regression training dominated execution time (99.7%).
- Data preparation and BoW conversion were secondary bottlenecks.

2- Optimization:
- OpenACC:
  - Parallelized logistic regression training using #pragma acc directives.
  - Utilized gang and vector parallelism to efficiently distribute workload.

- CUDA Python:
  - Accelerated data preparation and training using Numba.
  - Simplified implementation while maintaining performance gains.

- CUDA C:
  - Fine-tuned matrix operations and memory management.
  - Used atomic operations and dynamic memory allocation for thread safety.

3- Comparative Analysis:
  - Execution time reduced for the source code  7.99 seconds (CPU).
  - Execution time for OpenACC reduced to 0.27543 seconds ( native GPU).
  - Execution time for OpenACC reduced to 0.257 seconds (optimized GPU).
  - Execution time for Cuda C reduced to  1.026 seconds (native GPU).
  - Execution time for Cuda C reduced to 548 milliseconds (optimized GPU).
  -  Execution time for Cuda Python reduced to  1.026 seconds (native GPU).
  - Execution time for Cuda Python reduced to 548 milliseconds (optimized GPU).
  - Profiling results guided iterative optimization for balanced performance.


## Conclusion

This project demonstrates how GPU programming can drastically reduce computational time for NLP applications. The comparative analysis of OpenACC, CUDA Python, and CUDA C highlights the trade-offs between ease of programming and performance, showcasing GPU acceleration as a powerful tool for scalable NLP solutions.

