import os
import sys
from jinja2 import Environment, FileSystemLoader

from template import render_file, render_import

def build_template(customize_name, customize_class=None):
    assert customize_class is None

    template_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.join(os.getcwd(), "agent", customize_name)

    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    env = Environment(loader=FileSystemLoader(template_path))
    # Create meta template
    meta_template = env.get_template("meta_template.py")
    render_file(meta_template.render(customize_class=customize_class), \
                os.path.join(root_path, "meta_agent.py"))

    # Create evaulate template
    evaluate_template = env.get_template("evaluate_template.py")
    render_file(evaluate_template.render(customize_class=customize_class), \
                os.path.join(root_path, "evaluate_agent.py"))
    
    # Create search template
    search_template = env.get_template("search_template.py")
    render_file(search_template.render(customize_class=customize_class), \
                os.path.join(root_path, "search_agent.py"))

    # Create __init__ file
    init_template = env.get_template("init_template.py")
    render_file(init_template.render(customize_class=customize_class), \
            os.path.join(root_path, "__init__.py"))

    # Add import in agent __init__
    render_import(f"from .{customize_name}_agent import {customize_class}SearchAgent, {customize_class}EvaluateAgent", \
                os.path.join(root_path, "__init__.py"))
