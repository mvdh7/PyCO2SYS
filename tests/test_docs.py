# %%
from contextlib import redirect_stdout
from os import devnull, listdir
from pathlib import Path

import PyCO2SYS as pyco2  # noqa - this is assumed by all the docs


raise_errors = True  # usually should be True, can use False for manual testing


def test_docs():
    # This executes any code contained within a block starting ```python and
    # ending ``` in the online docs (any file in the docs folder ending ".md"),
    # just to check it can run without errors.
    # It doesn't necessarily mean that the code does what is advertised!
    docs_path = Path("docs")
    files = [f for f in listdir(docs_path) if f.endswith(".md")]
    for fname in files:
        with open(Path(docs_path, fname), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        is_code = False
        n_spaces = 0
        code_lines = ""
        mode = "v2"
        for line in lines:
            # Don't run code that's an example of how v1 worked
            # (should only be in v1_to_v2.md)
            if line == '=== "v1.8"':
                mode = "v1"
            elif line == '=== "v2.0"':
                mode = "v2"
            if mode == "v2":
                if is_code and line.strip() == "```":
                    is_code = False
                    n_spaces = 0
                if is_code:
                    code_lines += "\n" + line[n_spaces:]
                if line.strip() == "```python":
                    is_code = True
                    n_spaces = line.find("```python")
        try:
            with open(devnull, "w") as f, redirect_stdout(f):
                exec(code_lines)
        except Exception as e:
            print(f"ERROR in docs file {fname}")
            print(e)
            if raise_errors:
                raise Exception(e)


# test_docs()
