
## Key Components

1- Native CUDA Python Implementation (nativecudapython.py)
-  GPU Kernel: Implements logistic regression using global memory.
- Key Functions:
  - train_logistic_regression_gpu: Kernel for training logistic regression on GPU.
  - predict: Function for generating predictions based on trained weights and bias.
  - read_csv, tokenize, remove_stopwords, text_to_bow: Preprocessing utilities for data preparation.

- Features:
  - Utilizes global memory for all computations.
  - Basic GPU configuration with blocks and threads.
  - Atomic operations for weight updates.

- Performance:
  - Significant speedup over CPU implementations.
  - High computational efficiency.

2. Optimized CUDA Python Implementation (optcudapython.py)
- GPU Kernel Enhancements:
  - Utilizes shared memory for frequently accessed data (weights and biases).
  - Implements thread-level optimizations with cuda.shared.array.
  - Efficient atomic operations with strided updates.

- Key Functions:
    - logistic_regression_update_shared: Optimized kernel using shared memory.
    - initialize_weights, predict, read_csv, preprocess_texts: Utilities for weight initialization and preprocessing.

- Features:
    - Shared memory reduces global memory latency.
    - Thread synchronization ensures consistent updates.
    - Optimal grid and block configurations for better performance.

- Performance:
   - Improved execution time over the native version.
   - Enhanced GPU utilization and reduced overhead.
 
Enhancements in Optimized Version

1- Memory Management:
- Moved frequently accessed data (weights and biases) to shared memory.
- Reduced global memory latency, enhancing overall performance.

2- Thread Synchronization:
- Added cuda.syncthreads() to ensure consistent weight and bias updates.
- Minimized race conditions and improved data integrity.

3- Grid and Block Configuration:
- Optimized the number of blocks and threads per block for better GPU utilization.
- Calculated shared memory size dynamically based on features.

4- Atomic Operations:
- Improved atomic weight updates with strided access patterns.
- Reduced contention and enhanced throughput.

5- Execution Time:
- Achieved faster training time due to shared memory and kernel optimizations.
- Improved concurrency between threads.


## How to Run: 

Run the below commands: 
1- !python 02-python-code.py
2- !nsys profile -o profile_report python 02-python-code.py

