<<<<<<< HEAD
import time
from prefixspan import PrefixSpan
import numpy as np
from collections import Counter
from collections import OrderedDict
import json

def constrained_global_remap(frequent_patterns, value_range=(0, 255), freq_map_start=0, length_filter=4):
    # Step 1: Extract items appearing in target patterns
    freq_items = OrderedDict()

    for _, pattern in frequent_patterns:
        if len(pattern) >= length_filter:
            for item in pattern:
                freq_items[item] = True

    # Step 2: Assign compact new IDs
    freq_remap = {}
    remapped_used = set()
    next_id = freq_map_start

    for item in freq_items:
        while next_id in remapped_used:
            next_id += 1
        freq_remap[item] = next_id
        remapped_used.add(next_id)

    # Step 3: Build global mapping table (keep other IDs unchanged as much as possible)
    full_remap = {}
    for item in range(value_range[0], value_range[1] + 1):
        if item in freq_remap:
            full_remap[item] = freq_remap[item]
        else:
            if item not in remapped_used:
                full_remap[item] = item
                remapped_used.add(item)
            else:
                new_id = item
                while new_id in remapped_used and new_id <= value_range[1]:
                    new_id += 1
                if new_id > value_range[1]:
                    raise ValueError("ID range insufficient to complete mapping!")
                full_remap[item] = new_id
                remapped_used.add(new_id)

    return full_remap


def apply_remap_to_patterns(frequent_patterns, remap_table):
    """
    Replace items in frequent patterns according to the mapping table.
    """
    remapped_patterns = []
    for support, pattern in frequent_patterns:
        new_pattern = [remap_table[item] for item in pattern]
        remapped_patterns.append((support, new_pattern))
    return remapped_patterns


def main():
    # Parameters
    min_pattern_length = 4  # Minimum pattern length
    data_name = "sunspot"  # Dataset name
    support = 33  # Minimum support (occurrence count)
    sample_ratio = 0.2  # Sampling ratio

    with open(f"./few-shot-lab/{data_name}/frequent_patterns_support{support}_sample{sample_ratio}.json", "r") as f:
        pattern_list = json.load(f)

    # Restore to [(support, pattern), ...] format
    frequent_patterns = [(entry["support"], entry["pattern"]) for entry in pattern_list]


    remap_table = constrained_global_remap(
        frequent_patterns,
        value_range=(0, 255),
        freq_map_start=0,
        length_filter=min_pattern_length
    )

    remapped_patterns = apply_remap_to_patterns(frequent_patterns, remap_table)

    # Output mapping table
    print("\nID Mapping Table (only frequent items with length >= 4 are considered):")
    for orig, new in sorted(remap_table.items()):
        if orig != new:
            print(f"{orig} → {new}")

    i=0
    print("\nRemapped frequent patterns:")
    for supports, pattern in remapped_patterns:
        if len(pattern) >= 4:
            # Only output patterns with length >= 4
            print(f"Support={supports}, Pattern={pattern}")
            i += 1
            if i > 10:
                break

    # Save mapping table to remap_table.json
    save_path = f"./few-shot-lab/{data_name}/remap_table_support{support}_sample{sample_ratio}.json"
    with open(save_path, "w") as f:
        json.dump(remap_table, f, indent=4)

    print("\nMapping table has been saved to remap_table.json")



if __name__ == "__main__":
    main()
=======
import time
from prefixspan import PrefixSpan
import numpy as np
from collections import Counter
from collections import OrderedDict
import json

def constrained_global_remap(frequent_patterns, value_range=(0, 255), freq_map_start=0, length_filter=4):
    # Step 1: Extract items appearing in target patterns
    freq_items = OrderedDict()

    for _, pattern in frequent_patterns:
        if len(pattern) >= length_filter:
            for item in pattern:
                freq_items[item] = True

    # Step 2: Assign compact new IDs
    freq_remap = {}
    remapped_used = set()
    next_id = freq_map_start

    for item in freq_items:
        while next_id in remapped_used:
            next_id += 1
        freq_remap[item] = next_id
        remapped_used.add(next_id)

    # Step 3: Build global mapping table (keep other IDs unchanged as much as possible)
    full_remap = {}
    for item in range(value_range[0], value_range[1] + 1):
        if item in freq_remap:
            full_remap[item] = freq_remap[item]
        else:
            if item not in remapped_used:
                full_remap[item] = item
                remapped_used.add(item)
            else:
                new_id = item
                while new_id in remapped_used and new_id <= value_range[1]:
                    new_id += 1
                if new_id > value_range[1]:
                    raise ValueError("ID range insufficient to complete mapping!")
                full_remap[item] = new_id
                remapped_used.add(new_id)

    return full_remap


def apply_remap_to_patterns(frequent_patterns, remap_table):
    """
    Replace items in frequent patterns according to the mapping table.
    """
    remapped_patterns = []
    for support, pattern in frequent_patterns:
        new_pattern = [remap_table[item] for item in pattern]
        remapped_patterns.append((support, new_pattern))
    return remapped_patterns


def main():
    # Parameters
    min_pattern_length = 4  # Minimum pattern length
    data_name = "sunspot"  # Dataset name
    support = 33  # Minimum support (occurrence count)
    sample_ratio = 0.2  # Sampling ratio

    with open(f"./few-shot-lab/{data_name}/frequent_patterns_support{support}_sample{sample_ratio}.json", "r") as f:
        pattern_list = json.load(f)

    # Restore to [(support, pattern), ...] format
    frequent_patterns = [(entry["support"], entry["pattern"]) for entry in pattern_list]


    remap_table = constrained_global_remap(
        frequent_patterns,
        value_range=(0, 255),
        freq_map_start=0,
        length_filter=min_pattern_length
    )

    remapped_patterns = apply_remap_to_patterns(frequent_patterns, remap_table)

    # Output mapping table
    print("\nID Mapping Table (only frequent items with length >= 4 are considered):")
    for orig, new in sorted(remap_table.items()):
        if orig != new:
            print(f"{orig} → {new}")

    i=0
    print("\nRemapped frequent patterns:")
    for supports, pattern in remapped_patterns:
        if len(pattern) >= 4:
            # Only output patterns with length >= 4
            print(f"Support={supports}, Pattern={pattern}")
            i += 1
            if i > 10:
                break

    # Save mapping table to remap_table.json
    save_path = f"./few-shot-lab/{data_name}/remap_table_support{support}_sample{sample_ratio}.json"
    with open(save_path, "w") as f:
        json.dump(remap_table, f, indent=4)

    print("\nMapping table has been saved to remap_table.json")



if __name__ == "__main__":
    main()
>>>>>>> 8f5ca24 (Initial commit: add pretraining and inference modules)
