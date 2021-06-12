import sys
import numpy as np

from .base_lookup_table import calculate_model_efficient
from .block_builder import get_block
from .model import Model


def load_architecture(path_to_architecture):
    architecture = np.load(path_to_architecture)
    return architecture


def save_architecture(path_to_architecture, architecture):
    np.save(path_to_architecture, architecture)


def get_search_space_class(name):
    supernet_class = getattr(sys.modules[__name__], f"{name}Supernet")
    lookup_table_class = getattr(sys.modules[__name__], f"{name}LookUpTable")

    return supernet_class, lookup_table_class


from .fbnet import FBNetSSupernet, FBNetLSupernet, FBNetSLookUpTable, FBNetLLookUpTable
from .spos import SPOSSupernet, SPOSLookUpTable
from .proxylessnas import ProxylessNASSupernet, ProxylessNASLookUpTable
from .mobilefacenet import MobileFaceNetSupernet, MobileFaceNetLookUpTable
from .sgnas import SGNASSupernet, SGNASLookUpTable
