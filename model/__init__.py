import sys
import numpy as np

from .lookup_table_builder import LookUpTable, calculate_model_efficient
from .block_builder import get_block
from .model import Model


def load_architecture(path_to_architecture):
    architecture = np.load(path_to_architecture)
    return architecture


def save_architecture(path_to_architecture, architecture):
    np.save(path_to_architecture, architecture)


def get_supernet_class(name):
    return getattr(sys.modules[__name__], name)


from .fbnet import FBNetSSupernet, FBNetLSupernet
from .spos import SPOSSupernet
from .proxylessnas import ProxylessNASSupernet
from .mobilefacenet import MobileFaceNetSupernet
from .sgnas import SGNASSupernet
