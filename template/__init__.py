def render_file(render_content, render_file_path):
    with open(render_file_path, "w") as f:
        f.write(render_content)

def render_import(render_content, render_file_path):
    with open(render_file_path, "a") as f:
        f.writelines(render_content)

from . import agent, criterion, dataflow, search_space, search_strategy, training_strategy

