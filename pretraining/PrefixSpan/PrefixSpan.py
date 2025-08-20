<<<<<<< HEAD
from prefixspan import PrefixSpan
import time
import numpy as np
from collections import OrderedDict
import random
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
    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "ETTh1/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = timex

    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "ETTh2/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = np.concatenate((sequences, timex), axis=0)
    print("sequences.shape:", sequences.shape)

    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "weather/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = np.concatenate((sequences, timex), axis=0)
    print("sequences.shape:", sequences.shape)

    min_support = 500   # Minimum support (occurrence count)

    sample_ratio = 0.1
    sample_sequences = random.sample(list(sequences), int(len(sequences) * sample_ratio))

    sample_sequences = np.array(sample_sequences)

    # Run PrefixSpan on the sample
    ps = PrefixSpan(sample_sequences)
    start = time.time()
    frequent_patterns = ps.frequent(minsup=min_support, closed=True, filter=lambda patt, sup: len(patt) >= 4)
    elapsed = time.time() - start

    # Format as a list of dictionaries
    pattern_list = [{"pattern": list(map(int, pattern)), "support": int(support)} for support, pattern in frequent_patterns]

    with open("frequent_patterns.json", "w") as f:
        json.dump(pattern_list, f, indent=2)

    print("Frequent patterns have been saved to frequent_patterns.json")

    token_support = defaultdict(int)

    for support, pattern in frequent_patterns:
        unique_tokens = set(pattern)
        for token in unique_tokens:
            token_support[token] += support

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
from collections import OrderedDict
import random
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
    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "ETTh1/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = timex

    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "ETTh2/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = np.concatenate((sequences, timex), axis=0)
    print("sequences.shape:", sequences.shape)

    root_path = "E:/vscode/datapre/dataset/totem_data/"
    timex_file = root_path + "weather/Tin96_Tout96/train_x_codes.npy"
    timex = np.load(timex_file)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    sequences = np.concatenate((sequences, timex), axis=0)
    print("sequences.shape:", sequences.shape)

    min_support = 500   # Minimum support (occurrence count)

    sample_ratio = 0.1
    sample_sequences = random.sample(list(sequences), int(len(sequences) * sample_ratio))

    sample_sequences = np.array(sample_sequences)

    # Run PrefixSpan on the sample
    ps = PrefixSpan(sample_sequences)
    start = time.time()
    frequent_patterns = ps.frequent(minsup=min_support, closed=True, filter=lambda patt, sup: len(patt) >= 4)
    elapsed = time.time() - start

    # Format as a list of dictionaries
    pattern_list = [{"pattern": list(map(int, pattern)), "support": int(support)} for support, pattern in frequent_patterns]

    with open("frequent_patterns.json", "w") as f:
        json.dump(pattern_list, f, indent=2)

    print("Frequent patterns have been saved to frequent_patterns.json")

    token_support = defaultdict(int)

    for support, pattern in frequent_patterns:
        unique_tokens = set(pattern)
        for token in unique_tokens:
            token_support[token] += support

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
