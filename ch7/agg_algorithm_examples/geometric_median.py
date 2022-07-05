import numpy as np

# Aggregation function implementation

def geometric_median_aggregate(parameter_vectors, epsilon):
    vector_shape = parameter_vectors[0].shape
    vector_buffer = list(v.flatten() for v in parameter_vectors)

    prev_median = np.zeros(vector_buffer[0].shape)
    
    delta = np.inf

    vector_matrix = np.vstack(vector_buffer)

    while (delta > epsilon):
        dists = np.sqrt(np.sum((vector_matrix - prev_median[np.newaxis, :])**2, axis=1))
        curr_median = np.sum(vector_matrix / dists[:, np.newaxis], axis=0) / np.sum(1 / dists)
        delta = np.linalg.norm(curr_median - prev_median)
        prev_median = curr_median

    return prev_median.reshape(vector_shape)

def mean_aggregate(parameter_vectors):
    return np.mean(parameter_vectors, axis=0)

# Draw example parameter vectors from normal distribution

n = 10
d = 15

means = np.random.uniform(-10,10,d)

parameter_vectors = [np.random.normal(loc=means, size=d) for i in range(n)]

mean_agg_vector = mean_aggregate(parameter_vectors)

# Make the parameters of one model random noise

parameter_vectors[0] = np.random.uniform(-500,500,d)

mod_agg_vector = geometric_median_aggregate(parameter_vectors, epsilon=1e-5)
mod_mean_agg_vector = mean_aggregate(parameter_vectors)

print("Input vectors:")

for v in parameter_vectors:
    print(v)

print("\nTrue mean parameter vector (prior to replacing first model parameters with uniform noise):")
print(mean_agg_vector)

print("\nAggregate vector after replacement(mean):")
print(mod_mean_agg_vector)

print("\nAggregate parameter vector after replacement (geometric median):")
print(mod_agg_vector)