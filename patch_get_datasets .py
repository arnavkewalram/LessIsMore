import get_datasets

# Save the original get_dataset function
_original_get_dataset = get_datasets.get_dataset

def get_dataset_with_fraction(name: str, split: str, silent: bool = False, cache_dir: str = None, **kwargs):
    """Enhanced get_dataset that supports 'dataset:fraction' syntax"""
    if ':' in name:
        # Handle fractional dataset specification like "boolq:0.2"
        base_name, fraction = name.split(':')
        kwargs['data_fraction'] = float(fraction)
        name = base_name

    return _original_get_dataset(name, split, silent=silent, cache_dir=cache_dir, **kwargs)

# Replace the original function
get_datasets.get_dataset = get_dataset_with_fraction
print("✅ Patched get_dataset to support fractional datasets")
