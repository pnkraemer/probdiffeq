"""Generate API docs from the source."""

import pathlib


def main(
    src="probdiffeq",
    target="docs/API_documentation/",
    path_skip=("backend/*", "util/*"),
    nested_modules=("probdiffeq", "ivpsolve"),
):
    """Create an automatic API documentation.

    Recursively step through the src and
    turn each module into a mkdocstrings-compatible markdown file.
    """
    # Read the arguments
    path_source = pathlib.Path(src)
    path_target = pathlib.Path(target)

    path_skip = [
        *path_skip,
        *[f"{nest}.py" for nest in nested_modules],
        *[f"_{nest}/*" for nest in nested_modules],
    ]
    # Make the target directory unless it exists
    path_target.mkdir(parents=True, exist_ok=True)

    # Loop recursively through the source
    for path in sorted(path_source.rglob("*.py")):
        # Skip "private" modules and selected directories (e.g., backend/*)
        if path.name[0] != "_" and not any(path.match(s) for s in path_skip):
            # Construct the API_documentation filename and contents
            p_as_module = path_as_module(path)
            p_as_markdown = path_as_markdown_file(path, path_source)

            # Open the markdown file and write content
            with open(f"{path_target}/{p_as_markdown}.md", "w") as file:
                content = f"# {p_as_module} \n\n:::{p_as_module}"
                file.write(content)

    # Loop over the probdiffeq directory

    for nested in nested_modules:
        path_source = pathlib.Path(src) / f"_{nested}"
        (path_target / nested).mkdir(parents=True, exist_ok=True)

        with open(f"{path_target}/{nested}/.pages", "w") as file:
            file.write(f"title: probdiffeq.{nested}")

        # Read all submodules automatically and add them to the content
        for path in sorted(path_source.rglob("*.py")):
            if path.name[0] != "_":
                header = path.name.lower()
                header = header.replace(".py", "")
                header = header.replace("_and_", " & ")
                header = header.replace("_via_", ": ")
                header = header.replace("_", " ")
                header = header.capitalize()
                header = header.replace("Ssm", "SSM")
                p_as_module = path_as_module(path)
                content = f"""
# {header}

:::{p_as_module}
"""
                with open(f"{path_target}/{nested}/{header.lower()}.md", "w") as file:
                    file.write(content)


def path_as_module(p: pathlib.Path) -> str:
    """Turn a file-path to a python-module-like string."""
    p_as_string = str(p)
    return p_as_string.replace(".py", "").replace("/", ".")


def path_as_markdown_file(p: pathlib.Path, path_source: pathlib.Path) -> str:
    """Turn a file-path to a markdown-file-name."""
    return p.relative_to(path_source).stem


if __name__ == "__main__":
    main()
