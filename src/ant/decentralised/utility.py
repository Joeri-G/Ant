def k_highest_indices(lst, k):
    # Sort (index, value) pairs by value descending, take top k indices
    return [i for i, v in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:k]]
