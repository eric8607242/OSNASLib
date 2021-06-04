import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file

def build_template(customize_name):
    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "search_strategy")

    env = Environment(loader=FileSystemLoader(template_path))
    # Create criterion template
    meta_template = env.get_template("template.py")
    render_file(meta_template.render(customize_name=customize_name), os.path.join(root_path, f"{customize_name}.py"))
