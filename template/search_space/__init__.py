import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file, render_import

def build_interface(customize_name, customize_class=None):
    assert customize_class is not None

    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "search_space", customize_name)

    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    env = Environment(loader=FileSystemLoader(template_path))
    # Create supernet interface
    meta_template = env.get_template("supernet_template.py")
    render_file(meta_template.render(customize_class=customize_class), os.path.join(root_path, f"{customize_name}_supernet.py"))

    # Create model interface
    model_template = env.get_template("model_template.py")
    render_file(model_template.render(customize_class=customize_class), os.path.join(root_path, f"{customize_name}_model.py"))

    # Create lookup table interface
    lookup_table_template = env.get_template("lookup_table_template.py")
    render_file(lookup_table_template.render(customize_class=customize_class), os.path.join(root_path, f"{customize_name}_lookup_table.py"))
    
    # Create __init__ file
    render_file("", os.path.join(root_path, "__init__.py"))
    init_template = env.get_template("init_template.py")
    render_file(init_template.render(customize_name=customize_name, customize_class=customize_class), os.path.join(root_path, "__init__.py"))

    # Add import in model __init__
    render_import(f"from .{customize_name} import {customize_class}Supernet, {customize_class}LookUpTable, {customize_class}Model", os.path.join("search_space", "__init__.py"))
