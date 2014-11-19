import random


def reservoir(data, seed, amt):
    random.seed(seed)
    sample = []
    for i, line in enumerate(data):
        if i < amt:
            sample.append(line)
        elif i >= amt and random.random() < amt / float(i+1):
            replace = random.randint(0, len(sample)-1)
            sample[replace] = line

    return sample
