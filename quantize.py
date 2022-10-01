# https://gist.github.com/aleju/eb75fa01a1d5d5a785cf
def quantize(val, to_values):
    """Quantize a value with regards to a set of allowed values.
    
    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(43, [0, 10, 20, 30]) -> 30
    
    Note: function doesn't assume to_values to be sorted and
    iterates over all values (i.e. is rather slow).
    
    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    return best_match