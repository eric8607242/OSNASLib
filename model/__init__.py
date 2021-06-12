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

def get_model_class(name):
    return getattr(sys.modules[__name__], f"{name}Model")


from .fbnet import FBNetSSupernet, FBNetLSupernet, FBNetSLookUpTable, FBNetLLookUpTable, FBNetSModel, FBNetLModel
from .spos import SPOSSupernet, SPOSLookUpTable, SPOSModel
from .proxylessnas import ProxylessNASSupernet, ProxylessNASLookUpTable, ProxylessNASModel
from .mobilefacenet import MobileFaceNetSupernet, MobileFaceNetLookUpTable, MobileFaceNetModel
from .sgnas import SGNASSupernet, SGNASLookUpTable, SGNASModel
