import numpy as np
import hashlib
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy


def to_key(G):
    """
    Translates list or numpy array into hashable form.
    """
    G = np.array(G, dtype=np.int16)
    G = np.sort(G)
    key = G.tobytes()
    return key


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
	return np.argwhere(arr == np.array(search_list).reshape(len(search_list), 1))[:, 1]


def calculate_hash(args: DictConfig):
	args_copy = deepcopy(args)
	args_copy.exp.pop('mlflow_uri')
	return hashlib.md5(str(args_copy).encode()).hexdigest()

