import math
import pickle
import lmdb
import torch
from tqdm.auto import tqdm

# Core math (NaN-safe)
@torch.no_grad()
def _combine_stats(n1, m1, s1, n2, m2, s2):
    """
    Combine per-feature stats from two groups, NaN-agnostic.

    Inputs are [F]-shaped tensors:
      n*: counts (float or long ok; promoted to float)
      m*: means
      s*: sum of squared deviations (about each group's own mean)

    Returns (n_total, m_total, s_total)
    """
    n1 = n1.to(dtype=torch.float32)
    n2 = n2.to(dtype=torch.float32)
    n = n1 + n2

    # Avoid 0/0 using masking
    delta = m2 - m1
    m = m1 + torch.where(n > 0, delta * (n2 / n), torch.zeros_like(delta))
    s = s1 + s2 + torch.where(n > 0, delta.pow(2) * (n1 * n2 / n), torch.zeros_like(delta))
    return n, m, s

@torch.no_grad()
def _batch_stats_ignore_nan(x: torch.Tensor):
    """
    Compute per-feature stats for a batch x [T, F] with NaNs.

    Returns:
      k: valid count per feature [F] (float32)
      mean: nanmean per feature [F]
      s: sum of squared deviations around batch mean [F]
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    # Ensure float
    if not x.is_floating_point():
        x = x.float()

    valid = ~torch.isnan(x)
    k = valid.sum(dim=0)  # [F]
    # Replace NaNs with 0 for sums/SOS
    x0 = torch.where(valid, x, torch.zeros_like(x))
    sum_ = x0.sum(dim=0)
    sumsq = (x0 * x0).sum(dim=0)

    k_pos = k > 0
    mean = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    mean[k_pos] = sum_[k_pos] / k[k_pos]

    s = torch.zeros_like(mean)
    s[k_pos] = sumsq[k_pos] - (sum_[k_pos] ** 2) / k[k_pos]
    return k.to(torch.float32), mean, s

# LMDB readers

def _iter_lmdb_cursor_filtered(env, wanted_bytes_set):
    """
    Iterate (k, v) over LMDB using a cursor, yielding only keys in wanted_bytes_set.
    wanted_bytes_set: set of bytes keys (or None to yield all).
    """
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for k, v in cur:
            if (wanted_bytes_set is None) or (k in wanted_bytes_set):
                yield k, v

def _iter_lmdb_get_by_keys(env, key_bytes_list):
    """
    Iterate values by explicit key list (useful for multiprocessing shards).
    """
    with env.begin(write=False) as txn:
        for k in key_bytes_list:
            v = txn.get(k)
            if v is not None:
                yield k, v

# Worker logic (shared by single & multi)

def _process_stream(value_stream,
                    pickled_value_field: str = 'value',
                    show_progress: bool = True):
    """
    Consume (k, v) pairs where v is a pickled dict containing 'value' -> [T, F] (torch or numpy).
    Returns per-feature (n, m, s) tensors.
    """
    n = m = s = None

    iterator = tqdm(value_stream, desc="Scanning", unit="sample") if show_progress else value_stream
    for _, v in iterator:
        obj = pickle.loads(v)
        x = obj[pickled_value_field]  # [T, F], may contain NaNs

        # Ensure tensor on CPU
        x = torch.as_tensor(x)
        if x.ndim != 2:
            raise ValueError(f"Expected [time, features], got shape {tuple(x.shape)}")

        if m is None:
            F = x.shape[1]
            dev = torch.device('cpu')
            dtype = torch.float32
            n = torch.zeros(F, dtype=dtype, device=dev)
            m = torch.zeros(F, dtype=dtype, device=dev)
            s = torch.zeros(F, dtype=dtype, device=dev)

        k, mb, sb = _batch_stats_ignore_nan(x)
        n, m, s = _combine_stats(n, m, s, k, mb, sb)

    return n, m, s

def normaliser(
    listfile,
    *,
    lmdb_path: str,
    split_col: str = 'split',
    key_col: str = 'RecordID',
    pickled_value_field: str = 'value',
    use_cursor: bool = True,
    num_workers: int = 0,
    progress: bool = True,
    std_min_count: int = 2,
    set_unseen_to_nan: bool = False,
):
    """
    Compute mean/std per feature across all TRAIN samples in an LMDB, ignoring NaNs.

    Parameters
    ----------
    listfile : pandas.DataFrame
        Must include `split_col` (e.g., 'train') and `key_col` (patientunitstayid).
    lmdb_path : str
        Path to the LMDB environment.
    split_col : str
        Column with split labels ('train', 'val', ...).
    key_col : str
        Column with unique key values that correspond to LMDB keys.
    pickled_value_field : str
        Field name inside the pickled dict that contains the [T, F] array/tensor.
    use_cursor : bool
        If True, scan the DB with a cursor and filter (fast, sequential I/O).
        If False, do txn.get() per key (useful when multiprocessing shards by keys).
    num_workers : int
        If > 0, use multiprocessing with this many worker processes (reader-only).
        Each worker opens the LMDB in read-only mode and reduces its shard.
    progress : bool
        Show progress bars.
    std_min_count : int
        Minimum per-feature count to compute an unbiased std (default 2).
        Below this, std will be 0 or NaN (see set_unseen_to_nan).
    set_unseen_to_nan : bool
        If True, features with n==0 get mean/std = NaN (instead of 0).

    Returns
    -------
    dict with:
      'mean': torch.FloatTensor [F]
      'std' : torch.FloatTensor [F]
      'count': torch.FloatTensor [F]
    """
    # Build the set/list of wanted keys (bytes)
    train_keys = listfile.loc[listfile[split_col] == 'train', key_col].astype(str).tolist()
    wanted_bytes = [k.encode('utf-8') for k in train_keys]
    wanted_set = set(wanted_bytes) if use_cursor else None

    # Single-process path
    if num_workers <= 0:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=128)
        try:
            if use_cursor:
                stream = _iter_lmdb_cursor_filtered(env, wanted_set)
            else:
                stream = _iter_lmdb_get_by_keys(env, wanted_bytes)

            n, m, s = _process_stream(stream, pickled_value_field, show_progress=progress)
        finally:
            env.close()

    else:
        # Multiprocessing readers
        import multiprocessing as mp

        shards = []
        if use_cursor:
            per_key_lists = _split_list(wanted_bytes, num_workers)
            shards = per_key_lists
            cursor_per_worker = False
        else:
            shards = _split_list(wanted_bytes, num_workers)
            cursor_per_worker = False  # per-key mode

        # Worker function
        def _worker(path, key_bytes_list, pickled_field, show_prog):
            env_w = lmdb.open(path, readonly=True, lock=False, readahead=True, max_readers=128)
            try:
                stream = _iter_lmdb_get_by_keys(env_w, key_bytes_list) if not cursor_per_worker \
                    else _iter_lmdb_cursor_filtered(env_w, set(key_bytes_list))
                n_w, m_w, s_w = _process_stream(stream, pickled_field, show_progress=False)
                # Move to CPU float32 tensors (already are) and return via Pipe
                return (n_w.numpy(), m_w.numpy(), s_w.numpy())
            finally:
                env_w.close()

        with mp.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        _mp_wrapper,
                        [(lmdb_path, shard, pickled_value_field) for shard in shards]
                    ),
                    total=len(shards),
                    desc="Workers",
                    disable=not progress
                )
            )

        n = m = s = None
        for n_np, m_np, s_np in results:
            n_w = torch.from_numpy(n_np)
            m_w = torch.from_numpy(m_np)
            s_w = torch.from_numpy(s_np)
            if m is None:
                n, m, s = n_w, m_w, s_w
            else:
                n, m, s = _combine_stats(n, m, s, n_w, m_w, s_w)

    F = m.shape[0] if m is not None else 0
    if F == 0:
        return {'mean': torch.tensor([]), 'std': torch.tensor([]), 'count': torch.tensor([])}

    std = torch.zeros_like(m)
    if std_min_count <= 1:
        mask = n > 0
        std[mask] = torch.sqrt(s[mask] / n[mask])
    else:
        mask = n >= float(std_min_count)
        std[mask] = torch.sqrt(s[mask] / (n[mask] - 1.0))

    if set_unseen_to_nan:
        unseen = n == 0
        m[unseen] = float('nan')
        std[unseen] = float('nan')

    return {'mean': m, 'std': std, 'count': n}

# Helpers

def _split_list(seq, parts):
    L = len(seq)
    if parts <= 1 or L == 0:
        return [seq]
    size = math.ceil(L / parts)
    return [seq[i:i+size] for i in range(0, L, size)]

def _mp_wrapper(args):
    # Helper so Pool.imap_unordered can pickle callables cleanly
    lmdb_path, shard, pickled_field = args
    return _mp_worker_body(lmdb_path, shard, pickled_field)

def _mp_worker_body(lmdb_path, key_bytes_list, pickled_value_field):
    env_w = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=128)
    try:
        stream = _iter_lmdb_get_by_keys(env_w, key_bytes_list)
        n_w, m_w, s_w = _process_stream(stream, pickled_value_field, show_progress=False)
        return (n_w.numpy(), m_w.numpy(), s_w.numpy())
    finally:
        env_w.close()