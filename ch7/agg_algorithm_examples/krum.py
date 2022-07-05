import numpy as np

# Aggregation function implementation

def krum_aggregate(parameter_vectors, f, use_mean=False):
    num_vectors = len(parameter_vectors)

    filtered_size = max(1, num_vectors-f-2)

    scores = np.zeros(num_vectors)

    for i in range(num_vectors):
        distances = np.zeros(num_vectors)
        for j in range(num_vectors):
            distances[j] = np.linalg.norm(parameter_vectors[i] - parameter_vectors[j])
        scores[i] = np.sum(np.sort(distances)[:filtered_size])

    if (use_mean):
        idx = np.argsort(scores)[:filtered_size]
        return np.mean(np.stack(parameter_vectors)[idx], axis=0)
    else:
        idx = np.argmin(scores)
        return parameter_vectors[idx]

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

mod_agg_vector = krum_aggregate(parameter_vectors, f=int(n/4), use_mean=False)
mod_agg_vector_alt = krum_aggregate(parameter_vectors, f=int(n/4), use_mean=True)
mod_mean_agg_vector = mean_aggregate(parameter_vectors)

print("Input vectors:")

for v in parameter_vectors:
    print(v)

print("\nTrue mean parameter vector (prior to replacing first model parameters with uniform noise):")
print(mean_agg_vector)

print("\nAggregate vector after replacement(mean):")
print(mod_mean_agg_vector)

print(f"\nAggregate parameter vector after replacement (krum - single selection, f={int(n/4)}):")
print(mod_agg_vector)

print(f"\nAggregate parameter vector after replacement (krum - trimmed mean, f={int(n/4)}):")
print(mod_agg_vector_alt)