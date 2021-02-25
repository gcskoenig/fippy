import numpy as np
import hashlib
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from collections.abc import Iterable
import random
import mlflow
import itertools
import pandas as pd


def id_to_ix(id, ids):
    ids = np.unique(ids)
    ix = np.where(id == ids)[0][0]
    return ix

def fnames_to_key(fnames):
    fnames = sorted(set(deepcopy(fnames)))
    onestr = '|'.join(fname for fname in fnames)
    return hashlib.md5(str(onestr).encode()).hexdigest()

def to_key(G):
    """
    Translates list or numpy array into hashable form.
    """
    G = np.array(G, dtype=np.int16)
    G = np.sort(G)
    key = G.tobytes()
    return key

def create_multiindex(ns, vss):
    """Given names and values (ordered), create a
    pandas multiindex with all possible combinations
    following the ordering given in the respective
    lists or arrays
    """
    tuples = itertools.product(*vss)
    tuples = list(tuples)
    index = pd.MultiIndex.from_tuples(tuples, names=ns)
    return index

def ix_to_desc(j, base='X'):
    return '{}_{}'.format(base, j)


def ixs_to_desc(J, base='X'):
    descs = [ix_to_desc(jj, base=base) for jj in J]
    return descs


def rfi_desc(G, fs_names=None):
    """Generates string describing an Explainer
    Attributes:
        G: relative feature set
    """
    if fs_names is None:
        fs_names = ixs_to_desc(G)
    fs_names = np.array(fs_names)
    G_names = ','.join(fs_names[G])
    desc = '$RFI^{{{0}}}$'.format(G_names)
    return desc


def search_nonsorted(arr, search_list):
    if len(arr) == 0 or len(search_list) == 0:
        return np.array([])
    sl_reshape = np.array(search_list).reshape(len(search_list), 1)
    return np.argwhere(arr == sl_reshape)[:, 1]


def calculate_hash(args: DictConfig):
    args_copy = deepcopy(args)
    args_copy.exp.pop('mlflow_uri')
    return hashlib.md5(str(args_copy).encode()).hexdigest()


def flatten_gen(ls):
    for el in ls:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def flatten(ls):
    return list(flatten_gen(ls))


def sample_partial(partial_ordering):
    """ sample ordering from partial ordering

    Args:
        partial_ordering: [1, (2, 4), 3]

    Returns:
        ordering, np.array
    """
    ordering = []
    for elem in partial_ordering:
        if type(elem) is int:
            ordering.append(elem)
        elif type(elem) is tuple:
            perm = list(elem)
            random.shuffle(perm)
            ordering = ordering + perm
        else:
            raise RuntimeError('Element neither int nor tuple')
    return np.array(ordering)


def check_existing_hash(args: DictConfig, exp_name: str) -> bool:
    if args.exp.check_exisisting_hash:
        args.hash = calculate_hash(args)

        ids = mlflow.get_experiment_by_name(exp_name).experiment_id
        existing_runs = mlflow.search_runs(
            filter_string=f"params.hash = '{args.hash}'",
            run_view_type=mlflow.tracking.client.ViewType.ACTIVE_ONLY,
            experiment_ids=ids
        )
        return len(existing_runs) > 0
