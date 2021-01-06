from config_file.supernet_config import *
from utils.supernet import Supernet

a, b = get_supernet_cfg("proxylessnas", 10)
print(Supernet(a, b, 10, "a"))
