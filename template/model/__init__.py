import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file, render_import

def build_template(customize_name, customize_class=None):
    assert customize_class is not None

    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "model", customize_name)

    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    env = Environment(loader=FileSystemLoader(template_path))
    # Create meta template
    meta_template = env.get_template("template.py")
    render_file(meta_template.render(customize_class=customize_class), os.path.join(root_path, f"{customize_name}_supernet.py"))

    # Create __init__ file
    render_file("", os.path.join(root_path, "__init__.py"))
    init_template = env.get_template("init_template.py")
    render_file(meta_template.render(customize_name=customize_name, customize_class=customize_class), os.path.join(root_path, f"{customize_name}_supernet.py"))

    # Add import in model __init__
    render_import(f"from .{customize_name} import {customize_class}Supernet", os.path.join("model", "__init__.py"))
