import numpy as np


def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b 

def entropy(signal, n_bins=50):


    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    
    # Create histogram (probability distribution)
    hist, bin_edges = np.histogram(signal, bins=n_bins, density=True)
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy
