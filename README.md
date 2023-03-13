# parallel-all-pairs-shortest-path
Solve the all pairs shortest path problem using blocked Floyd-Warshall algorithm, and further accelerate the program using CUDA.  
Rank (based on execution time):  
- CPU version (6/60)
- 1 GPU version (6/58)
- 2 GPU version (1/51)

## Implementation (full details in the report)
### CPU version (hw3-1)
used threading and vectorization to parallelize the computation in the program. 

### 1 GPU version (hw3-2)
To fully utilized the shared memory the `blocking factor` = 64.  
`block-size` = (32, 32), and to avoid bank conflict, for example thread 0 will be responsible for (0,0), (32,0), (0,32), and (32,32). Lastly,`max number of threads` = 1024.

### 2 GPU version (hw3-3)
used p2p communication method, phase 3 is divided to 2 GPU, while each GPU will recalculate the phase 1 and phase 2.
