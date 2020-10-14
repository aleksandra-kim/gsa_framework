import multiprocessing


class ComputationalResources:
    def __init__(self, memory=2, cpus=None, bytes_per_float=8):
        """Common and simple interface to know how much CPU and memory is available.

        ``memory`` is in GB."""
        self.memory = memory
        self.cpus = cpus or multiprocessing.cpu_count()

    # Extra utility functions related to splitting things into chunks, etc.
