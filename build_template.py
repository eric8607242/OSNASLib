import os
import sys
import argparse

import template

def main(template_type, customize_name):
    template_builder = getattr(getattr(template, template_type), "build_template")
    template_builder(customize_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tt", "--template-type", required=True, type=str)
    parser.add_argument("--customize-name", required=True, type=str)
    args = vars(parser.parse_args())

    main(args["template_type"], args["customize_name"])

