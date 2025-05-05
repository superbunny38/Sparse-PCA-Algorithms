# Sparse-PCA-Algorithms
Advanced Algorithms (25 Spring, Columbia University)

(R) Brute-force SPCA: This method searches over all possible subsets of features (i.e. support sets S  {1, ..., p}) of size at most k, and for each set, it computes the top eigenvalue of the submatrix AS . This gives the exact optimal solution to the sparse PCA problem but is computationally infeasible for large p and k.

(R) DSPCA Upper Bound: This method relaxes the SPCA problem into a convex semidefinite program. Instead of searching for a sparse vector directly, it introduces a positive semidefinite matrix variable and relaxes the rank and sparsity constraints. This relaxed problem can be solved efficiently using convex optimization tools. The result provides an upper bound on the variance that can be explained by any k-sparse principal component.

(C) Random Sampling Lower Bound: This method randomly selects subsets of features of size at most k, computes the largest eigenvalue of each corresponding submatrix AS , and reports the maximum across all samples. This gives a lower bound on the best possible variance a sparse component can explain, but since it relies on random sampling, the quality of the bound may vary.

(C) Greedy SPCA: This is a greedy heuristic algorithm that builds a support set S iteratively. Starting with an empty set, it adds at each step the feature j that leads to the largest increase in the top eigenvalue of the submatrix AS{j}. This process continues until |S| = k. Although this is more of a heuristic, it is computationally efficient, produces interpretable results, and often gives a strong lower bound of the SPCA objective.
