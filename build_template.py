import os
import sys
import argparse

import template

def main(template_type, customize_name, customize_class=None):
    template_builder = getattr(getattr(template, template_type), "build_template")
    template_builder(customize_name, customize_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tt", "--template-type", required=True, type=str)
    parser.add_argument("--customize-name", required=True, type=str)
    parser.add_argument("--customize-class", type=str)
    args = vars(parser.parse_args())

    main(args["template_type"], args["customize_name"], args["customize_class"])

