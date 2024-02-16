import os
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()



# Change this to match the file extension of your code
FILE_EXTENSION = ".py"
SRC_DIR = "yasfpy"
TODO_SRC = "docs/todo_temp.md"
TODO_DEST = "docs/todo.md"

def find_todo_comments(directory):
    todo_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(FILE_EXTENSION):
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip().startswith("# TODO"):
                            todo_list.append(
                                f"- [{file}:{i + 1}](https://github.com/AGBV/YASF/blob/main/{os.path.join(root, file)}#L{i + 1}){line.strip().replace('# TODO', '')}"
                            )
    return todo_list


def generate_todo_md(directory, output_file, source_file = None):
    todo_list = find_todo_comments(directory)
    todo = ""
    with open(source_file, "r") as f:
        todo = f.readlines()
        todo = "".join(todo)
    with open(output_file, "w") as f:
        f.write(todo)
        f.write("\n\n## Source List\n")
        f.write("**This is an auto-generated list of TODOs in the codebase.**\n\n")
        f.write("\n".join(todo_list))

src = Path(__file__).parent.parent
generate_todo_md(src / SRC_DIR, "docs/todo_gen.md", "docs/todo.md")