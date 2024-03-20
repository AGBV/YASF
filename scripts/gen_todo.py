import os
from pathlib import Path

import mkdocs_gen_files

# Change this to match the file extension of your code
FILE_EXTENSION = ".py"
SRC_DIR = "yasfpy"
TODO_SRC = "docs/todo_temp.md"
TODO_DEST = "docs/todo.md"


def find_todo_comments(directory):
    todo_list = []
    print(directory.parent)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(FILE_EXTENSION):
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip().startswith("# TODO"):
                            file_url = f"https://github.com/AGBV/YASF/blob/main/{os.path.join(root.replace(directory.parent.as_posix(), '').strip('/'), file)}#L{i + 1}"
                            todo_list.append(
                                f"- [{file}:{i + 1}]({file_url}){{:target='blank'}}{line.strip().replace('# TODO', '')}"
                            )
    return todo_list


def generate_todo_md(directory, output_file, source_file=None):
    todo_list = find_todo_comments(directory)
    todo = ""
    with open(source_file, "r") as f:
        todo = f.readlines()
        todo = "".join(todo)
    with mkdocs_gen_files.open(output_file, "w") as f:
        f.write(todo)
        f.write("\n\n## Source List\n")
        f.write("**This is an auto-generated list of TODOs in the codebase.**\n\n")
        f.write("\n".join(todo_list))


src = Path(__file__).parent.parent
generate_todo_md(src / SRC_DIR, "todo_gen.md", "docs/todo.md")
