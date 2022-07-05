import numpy as np

# Aggregation function implementation

def coordinate_median_aggregate(parameter_vectors):
    return np.median(parameter_vectors, axis=0)

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

mod_agg_vector = coordinate_median_aggregate(parameter_vectors)
mod_mean_agg_vector = mean_aggregate(parameter_vectors)

print("Input vectors:")

for v in parameter_vectors:
    print(v)

print("\nTrue mean parameter vector (prior to replacing first model parameters with uniform noise):")
print(mean_agg_vector)

print("\nAggregate vector after replacement(mean):")
print(mod_mean_agg_vector)

print("\nAggregate parameter vector after replacement (coordinate-wise median):")
print(mod_agg_vector)