import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file, render_import

def build_interface(customize_name, customize_class=None):
    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "dataflow")

    env = Environment(loader=FileSystemLoader(template_path))
    # Create criterion interface
    meta_template = env.get_template("template.py")
    render_file(meta_template.render(customize_name=customize_name), os.path.join(root_path, f"{customize_name}.py"))

    # Add import in dataflow __init__
    render_import(f"from .{customize_name} import get_{customize_name}_dataloader", os.path.join("dataflow", "__init__.py"))

