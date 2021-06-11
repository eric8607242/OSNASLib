import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file, render_import

def build_template(customize_name, customize_class=None):
    assert customize_class is not None

    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "training_strategy")

    env = Environment(loader=FileSystemLoader(template_path))
    # Create criterion template
    meta_template = env.get_template("template.py")
    render_file(meta_template.render(customize_class=customize_class), os.path.join(root_path, f"{customize_name}.py"))

    # Add import in training strategy __init__
    render_import(f"from .{customize_name} import {customize_class}Sampler", os.path.join("training_strategy", "__init__.py"))
