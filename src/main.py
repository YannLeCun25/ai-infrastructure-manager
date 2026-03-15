from scheduler import GPUScheduler

def main():
    cluster = GPUScheduler(num_gpus=8)
    print(cluster.allocate("ResNet-50-Training"))
    print(cluster.allocate("GPT-4-Inference"))

if __name__ == "__main__":
    main()
