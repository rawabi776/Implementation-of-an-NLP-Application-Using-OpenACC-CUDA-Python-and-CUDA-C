
## Enhancements in CUDA C Implementation:

1- Memory Management:

- Native Implementation:
   - Relies on Unified Memory for simplified memory allocation and synchronization between host and device.
   - Memory allocation for input data, labels, weights, and bias is managed automatically.
   - Trade-off: Unified Memory introduces page fault overhead and limits performance in high-throughput tasks.

- Optimized Implementation:
    - Uses Explicit Device Memory Management, allocating memory directly on the GPU with cudaMalloc and cudaMemcpy.
    - Memory transfers between host and device are fine-tuned for performance.
    - Trade-off: More complex to implement but reduces overhead from page faults.

2- Kernel Design:

- Native Implementation:
  - A single kernel is launched with basic parallelism for logistic regression.
  - Grid and Block Settings: Simple grid and block configurations with no consideration for stream concurrency.

- Optimized Implementation:

  - Enhances kernel execution by dynamically tuning the number of streams and assigning workloads to available Streaming Multiprocessors (SMs).
  -  Stream Utilization: Asynchronous data transfers and kernel launches reduce idle GPU time.
  -  Multiple CUDA streams enable overlapping computation and memory transfers.
3- Parallelism:

- Native Implementation:
    - Basic thread parallelism for processing samples in the kernel.

- Optimized Implementation:
  - Uses fine-grained atomic operations (atomicAdd) for weight and bias updates in parallel threads.
  - Overcomes thread contention by dividing samples across streams.

4- Profiling and Performance:

- Native Implementation:

  - Basic time measurement with CUDA Events to log execution time.

- Optimized Implementation:

   - Uses GPU device properties (e.g., number of SMs) to dynamically configure kernel and stream execution.
   - Includes comprehensive profiling to identify bottlenecks and optimize kernel performance.

5- Scalability:

- Native Implementation:
    - Simplistic approach suitable for smaller datasets but struggles with larger inputs due to Unified Memory overhead.

- Optimized Implementation:
    - Scales efficiently with larger datasets by leveraging streams and optimizing memory management.

## How to  Run 

1- !python 03-python-code.py
2- !nsys profile -o python_report python 03-python-code.py
