import os
import sys
import argparse

import template

def main(interface_type, customize_name, customize_class=None):
    interface_builder = getattr(getattr(template, interface_type), "build_interface")
    interface_builder(customize_name, customize_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--interface-type", required=True, type=str)
    parser.add_argument("--customize-name", required=True, type=str)
    parser.add_argument("--customize-class", type=str)
    args = vars(parser.parse_args())

    main(args["interface_type"], args["customize_name"], args["customize_class"])

