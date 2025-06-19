# Deduplication informed upsampling: Rehydration

## About rehydration
While deduplication in FineWeb2 is performed globally per language, we use the number of duplicates of each kept document + the filtering rates per duplicate cluster size to inform our upsampling approach.

For most languages we inspected, documents that had very few or way too many duplicates tend to be of lower quality, while documents with a number of duplicates relatively in "the middle" were of higher quality, which is shown by looking at the filtering rates by cluster size (as lower quality documents would, normally, be removed more often by our filtering pipeline than higher quality documents).

As such, for cluster sizes where the filtering rate was close to or higher than the global filtering rate, we do not upsample the documents at all; for the cluster size with the lowest filtering rate we repeat these document the maximum number of times (a configurable value, 10 in our experiments); and for anything in between we linearly interpolate between these 2 extremes.

## Pre-computed upsampling weights
In the [weights/](weights/) folder, we provide upsampling weights for each language for different maximum repetition counts (**3 to 10**, inclusive).

For instance, for up to 10 repetitions, [weights/up_to_10_reps.json](weights/up_to_10_reps.json) contains the following:

```
'ita_Latn': {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 7: 7, 8: 8, 11: 9, 17: 10, 26: 9, 31: 8, 46: 7, 52: 6, 63: 5, 75: 4, 96: 3, 118: 2, 123: 3, 125: 2, 145: 1}
```

This means that:
- documents that had no duplicates aren't repeated at all (weight=1)
- documents that had 1 duplicate are repeated 2 times (weight=3)
- ...
- documents that had 16-24 duplicates should be repeated 9 times (weight=10)
- etc

These weights inflate the `ita_Latn` subset to 965462863 documents (see [weights/resulting_ds_sizes.json](weights/resulting_ds_sizes.json)), from its original size of 238984437 (a 4x increase). Instead of naively feeding every single document to a model 4 times, rehydration allows us to repeat higher quality documents while still keeping diversity.

## Rehydrating a subset

See the following example datatrove block that, given weights as defined in the weights folder, rehydrates the dataset:

```python
class Rehydrater(PipelineStep):
    def __init__(self, upsampling_weights):
        self.upsampling_weights = upsampling_weights
        super().__init__()

    def run(self, data, rank: int = 0, world_size: int = 1):
        limits = set(self.upsampling_weights.keys())
        expanded_weights = [1]
        for i in range(1, max(limits) + 1):
            if i in limits:
                expanded_weights.append(self.upsampling_weights[i])
            else:
                expanded_weights.append(expanded_weights[-1])

        for doc in data:
            upsampling_weight = expanded_weights[-1] if doc.metadata["minhash_cluster_size"] >= len(expanded_weights) else expanded_weights[doc.metadata["minhash_cluster_size"]]
            for _ in range(upsampling_weight):
                yield doc
```

use with:

```python
    pipeline=[
        JsonlReader(path),
        Rehydrater({1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 7: 7, 8: 8, 11: 9, 17: 10, 26: 9, 31: 8, 46: 7, 52: 6, 63: 5, 75: 4, 96: 3, 118: 2, 123: 3, 125: 2, 145: 1}),
        DocumentTokenizer(...)
    ]
```

## Computing your own weights

The distribution of filtering rates by cluster size are available in [distributions/](distributions/).
We used the following code to compute upsampling weights:

```python
lang = "ita_Latn"
max_repetitions = 10

def compute_repetitions(sizes: np.ndarray, rates: np.ndarray, doc_counts: np.ndarray, global_rate: float, max_reps: int, window: int = 5) -> Tuple[dict, int]:
    """Compute repetitions for each cluster size based on removal rates."""
    # Identify clusters above global rate (will get 1 repetition)
    above_global = rates >= global_rate

    # Find minimum rate for interpolation
    min_rate = np.min(rates)

    # Calculate repetitions using linear interpolation
    reps = np.ones_like(rates, dtype=float)
    below_global = ~above_global
    if np.any(below_global):
        # Linear interpolation between 1 and max_reps based on rate
        reps[below_global] = 1 + (max_reps - 1) * (
                (rates[below_global] - global_rate) / (min_rate - global_rate)
        )

    # Apply smoothing
    kernel = np.ones(window) / window
    smoothed_reps = np.convolve(reps, kernel, mode='same')

    # Handle edges (use original values at the edges)
    half_window = window // 2
    smoothed_reps[:half_window] = reps[:half_window]
    smoothed_reps[-half_window:] = reps[-half_window:]

    # Round to nearest integer
    repetitions = np.round(smoothed_reps).astype(int)

    current_rep = repetitions[0]
    range_start = sizes[0]
    total_docs = 0

    # Dictionary for compact representation
    rep_dict = {}
    for i in range(len(sizes)):
        # Update total document count
        total_docs += doc_counts[i] * repetitions[i]

        # Check if we need to close current range
        if i == len(sizes) - 1 or repetitions[i + 1] != current_rep:
            # Add to dictionary
            rep_dict[int(range_start)] = int(current_rep)

            if i < len(sizes) - 1:
                range_start = sizes[i + 1]
                current_rep = repetitions[i + 1]
    
    return rep_dict, total_docs.item()


with open(f"distributions/{lang}.json", "rt") as f:
    data = json.load(f)

tail_threshold = data["tail_threshold"]
tail_removal_rate = data["tail_removal_rate"]
tail_filt_counts = data["tail_post_filtering_doc_counts"]
global_removal_rate = data["global_removal_rate"]

# Get the main distribution
sizes = np.array(data["cluster_sizes"] + [tail_threshold])
rates = np.array(data["cluster_removal_rates"] + [tail_removal_rate])
filt_counts = np.array(data["cluster_post_filtering_doc_counts"] + [tail_filt_counts])

upsampling_weights, resulting_total_docs = compute_repetitions(sizes, rates, filt_counts, global_removal_rate, max_repetitions)
```