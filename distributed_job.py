import ray

ray.init(address="auto")


@ray.remote
def compute(x):
    return x * x


futures = [compute.remote(i) for i in range(10)]
results = ray.get(futures)
print("Results:", results)
