import numpy as np
import matplotlib.pyplot as plt

def thresholding_spca(A, k):
    # Compute leading eigenvector
    eigvals, eigvecs = np.linalg.eigh(A)
    v1 = eigvecs[:, -1]  # Top eigenvector
    
    # Get top-k absolute value indices
    top_k_indices = np.argsort(np.abs(v1))[-k:]
    
    # Zero out all but top-k entries
    v_sparse = np.zeros_like(v1)
    v_sparse[top_k_indices] = v1[top_k_indices]
    
    # Normalize
    v_sparse /= np.linalg.norm(v_sparse)
    
    # Return explained variance
    return v_sparse.T @ A @ v_sparse

def main_thresholding_plot():
    np.random.seed(42)
    K = 30
    n_dims_list = [2, 5, 10, 20]
    results = {}

    for p in n_dims_list:
        results[p] = []
        D = np.random.randn(K, p)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        B = D @ D.T
        B /= np.linalg.norm(B, ord=2)  # Normalize spectral norm to 1

        for k in range(1, K + 1):
            val = thresholding_spca(B, k)
            if k > 1:
                val = max(val, results[p][-1])  # enforce monotonicity
            results[p].append(val)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.title("Thresholding SPCA\n(effect of number of dimensions)")
    for p in n_dims_list:
        plt.plot(range(1, K + 1), results[p], label=f"{p} dim")
    plt.xlabel("Sparsity level (k)")
    plt.ylabel("k-SPCA value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("thresholding_spca_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main_thresholding_plot()