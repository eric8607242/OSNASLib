import numpy as np

from .lookup_table_builder import LookUpTable
from .network_utils import get_block
from .supernet import Supernet
from .model import Model


def load_architecture(path_to_architecture):
    architecture = np.load(path_to_architecture)
    return architecture


def save_architecture(path_to_architecture, architecture):
    np.save(path_to_architecture, architecture)
