<<<<<<< HEAD
from prefixspan import PrefixSpan
import time
import numpy as np
from collections import defaultdict
import json

class TimedPrefixSpan(PrefixSpan):
    def __init__(self, sequences, max_patterns=100):
        super().__init__(sequences)
        self.count = 0
        self.max_patterns = max_patterns
        self.start = time.time()

    def _PrefixSpan__frequent(self, pattern, projected, freq, results):
        self.count += 1
        if self.count >= self.max_patterns:
            raise Exception("Early stopping for estimation")
        return super()._PrefixSpan__frequent(pattern, projected, freq, results)

if __name__ == "__main__":
    dataset_name = "us_births"
    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + f"{dataset_name}/Tin96_Tout96/test_x_codes.npy"
    timex = np.load(timex_file)  # [bs, compressed_time, sensors]
    print("timex.shape:", timex.shape)
    timex = timex.reshape(-1, timex.shape[1])  # [bs * sensors, compressed_time]

    sequences = timex
    print("sequences.shape:", sequences.shape)

    min_support = 33  # Minimum support (occurrence count)

    # Random sampling from raw data, e.g., take 10% of the data
    sample_ratio = 0.20
    index = int(len(sequences) * sample_ratio)
    sample_sequences = sequences[:index]  # Take the first index samples

    # Run PrefixSpan on the sample
    ps = PrefixSpan(sample_sequences)
    start = time.time()
    # ps.frequent(min_support * sample_ratio)  # Support should also be scaled down
    frequent_patterns = ps.frequent(minsup=min_support, closed=True, filter=lambda patt, sup: len(patt) >= 4)
    elapsed = time.time() - start

    # Linear extrapolation
    estimated_total_time = elapsed / sample_ratio
    print(f"Estimated total runtime is about: {estimated_total_time:.2f} seconds")

    # Format as a list of dictionaries
    pattern_list = [{"pattern": list(map(int, pattern)), "support": int(support)} for support, pattern in frequent_patterns]

    with open(f"./few-shot-lab/{dataset_name}/frequent_patterns_support{min_support}_sample{sample_ratio}.json", "w") as f:
        json.dump(pattern_list, f, indent=2)

    print(f"Frequent patterns have been saved to ./few-shot-lab/{dataset_name}/frequent_patterns_support{min_support}_sample{sample_ratio}.json")

    # Calculate the total support of each token
    token_support = defaultdict(int)

    for support, pattern in frequent_patterns:
        unique_tokens = set(pattern)  # Avoid duplicate counting in the same sequence (e.g., 6 in [6, 6, 7])
        for token in unique_tokens:
            token_support[token] += support

    # Sort (by support in descending order)
    sorted_support = sorted(token_support.items(), key=lambda x: -x[1])

    print("ID\tSupport")
    for token, support in sorted_support:
        print(f"{token}\t{support}")

    # Statistics of frequent pattern length distribution
    length_distribution = defaultdict(int)
    for support, pattern in frequent_patterns:
        length_distribution[len(pattern)] += 1
    sorted_length_distribution = sorted(length_distribution.items(), key=lambda x: x[0])
    print("\nFrequent pattern length distribution:")
    print("Length\tProportion")
    for length, count in sorted_length_distribution:
        print(f"{length}\t{count/len(frequent_patterns):.2%}")

    print("\nNumber of frequent patterns:", len(sorted_support))
=======
from prefixspan import PrefixSpan
import time
import numpy as np
from collections import defaultdict
import json

class TimedPrefixSpan(PrefixSpan):
    def __init__(self, sequences, max_patterns=100):
        super().__init__(sequences)
        self.count = 0
        self.max_patterns = max_patterns
        self.start = time.time()

    def _PrefixSpan__frequent(self, pattern, projected, freq, results):
        self.count += 1
        if self.count >= self.max_patterns:
            raise Exception("Early stopping for estimation")
        return super()._PrefixSpan__frequent(pattern, projected, freq, results)

if __name__ == "__main__":
    dataset_name = "us_births"
    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + f"{dataset_name}/Tin96_Tout96/test_x_codes.npy"
    timex = np.load(timex_file)  # [bs, compressed_time, sensors]
    print("timex.shape:", timex.shape)
    timex = timex.reshape(-1, timex.shape[1])  # [bs * sensors, compressed_time]

    sequences = timex
    print("sequences.shape:", sequences.shape)

    min_support = 33  # Minimum support (occurrence count)

    # Random sampling from raw data, e.g., take 10% of the data
    sample_ratio = 0.20
    index = int(len(sequences) * sample_ratio)
    sample_sequences = sequences[:index]  # Take the first index samples

    # Run PrefixSpan on the sample
    ps = PrefixSpan(sample_sequences)
    start = time.time()
    # ps.frequent(min_support * sample_ratio)  # Support should also be scaled down
    frequent_patterns = ps.frequent(minsup=min_support, closed=True, filter=lambda patt, sup: len(patt) >= 4)
    elapsed = time.time() - start

    # Linear extrapolation
    estimated_total_time = elapsed / sample_ratio
    print(f"Estimated total runtime is about: {estimated_total_time:.2f} seconds")

    # Format as a list of dictionaries
    pattern_list = [{"pattern": list(map(int, pattern)), "support": int(support)} for support, pattern in frequent_patterns]

    with open(f"./few-shot-lab/{dataset_name}/frequent_patterns_support{min_support}_sample{sample_ratio}.json", "w") as f:
        json.dump(pattern_list, f, indent=2)

    print(f"Frequent patterns have been saved to ./few-shot-lab/{dataset_name}/frequent_patterns_support{min_support}_sample{sample_ratio}.json")

    # Calculate the total support of each token
    token_support = defaultdict(int)

    for support, pattern in frequent_patterns:
        unique_tokens = set(pattern)  # Avoid duplicate counting in the same sequence (e.g., 6 in [6, 6, 7])
        for token in unique_tokens:
            token_support[token] += support

    # Sort (by support in descending order)
    sorted_support = sorted(token_support.items(), key=lambda x: -x[1])

    print("ID\tSupport")
    for token, support in sorted_support:
        print(f"{token}\t{support}")

    # Statistics of frequent pattern length distribution
    length_distribution = defaultdict(int)
    for support, pattern in frequent_patterns:
        length_distribution[len(pattern)] += 1
    sorted_length_distribution = sorted(length_distribution.items(), key=lambda x: x[0])
    print("\nFrequent pattern length distribution:")
    print("Length\tProportion")
    for length, count in sorted_length_distribution:
        print(f"{length}\t{count/len(frequent_patterns):.2%}")

    print("\nNumber of frequent patterns:", len(sorted_support))
>>>>>>> 8f5ca24 (Initial commit: add pretraining and inference modules)
