import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

mappings_path = os.path.join(os.path.dirname(__file__), "mappings.json")


def _parse_from_imports(
    mappings: Dict[str, str],
    installed_modules: List[str],
    line_idx: int,
    lines: List[str],
    verbose: bool = False,
):
    new_lines = []
    new_installs = []
    imported_modules = []
    parsing_modules = False
    skipped_lines = 0

    # Optimize installed_modules checks by using a set for O(1) membership tests.
    installed_modules_set = set(installed_modules)

    # Precompute and cache the replacement string for "llama_index"
    LI_REPLACEMENT = "llama_index.core"

    length_lines = len(lines)
    i = line_idx

    while i < length_lines:
        line = lines[i]
        skipped_lines += 1
        if "from " in line:
            import_stripped = line.strip()
            part_import = import_stripped.split(" import ")
            imported_part = part_import[-1].strip()
            imported_modules = [line, imported_part]
            if imported_part.startswith("("):
                imported_modules[-1] = []
                parsing_modules = True
            else:
                imported_modules = [line, imported_part.split(", ")]
        

        if parsing_modules:
            if ")" in line:
                parsing_modules = False
            elif "(" not in line:
                imported_modules[-1].append(line.strip().replace(",", ""))

        if not parsing_modules and imported_modules:
            # flatten to module names
            imported_module_names = [x.strip() for x in imported_modules[-1]]
            new_imports_dict = {}

            for module in imported_module_names:
                if module in mappings:
                    new_import_parent = mappings[module]
                else:
                    print(f"Module not found: {module}\nSwitching to core")
                    # get back the llama_index module that's being imported.
                    new_import_parent = (
                        imported_modules[0].split(" import ")[0].split("from ")[-1]
                    )
                    # if the parent contains `llama_index.core` already, then skip
                    if "llama_index.core" not in new_import_parent:
                        new_import_parent = new_import_parent.replace(
                            "llama_index", LI_REPLACEMENT
                        )
                # Aggregate modules per parent
                if new_import_parent not in new_imports_dict:
                    new_imports_dict[new_import_parent] = [module]
                else:
                    new_imports_dict[new_import_parent].append(module)
            
            for new_import_parent, module_list in new_imports_dict.items():
                # Precompute install parent
                new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
                # Installed check using set (O(1))
                if new_install_parent not in installed_modules_set:
                    # Optimize overlap check by searching, not scanning full list
                    found_overlap = False
                    for x in installed_modules_set:
                        if x in new_install_parent:
                            found_overlap = True
                            break
                    if not found_overlap:
                        installed_modules.append(new_install_parent)
                        installed_modules_set.add(new_install_parent)
                        new_installs.append(f"%pip install {new_install_parent}\n")
                modules_joined = ", ".join(module_list)
                new_lines.append(f"from {new_import_parent} import {modules_joined}\n")

            # reset flags/arrays for next round!
            parsing_modules = False
            imported_modules = []

            return new_lines, new_installs, installed_modules, skipped_lines

        elif not parsing_modules:
            new_lines.append(line)
        i += 1

    return new_lines, new_installs, installed_modules, skipped_lines


def _parse_hub_downloads(
    mappings: Dict[str, str],
    installed_modules: List[str],
    line: str,
):
    regex = r"download_loader\([\"']([A-Z,a-z]+)[\"'][\s,a-z,A-Z,_=]*\)|download_tool\([\"']([a-z,A-Z]+)[\"'][A-Z,a-z,\s,_=]*\)"
    result = re.search(regex, line)
    new_lines = []
    new_installs = []
    if result:
        tool, reader = result.groups()
        module = tool if tool else reader
        if module in mappings:
            new_import_parent = mappings[module]
            new_lines.append(f"from {new_import_parent} import {module}\n")
            new_install_parent = new_import_parent.replace(".", "-").replace("_", "-")
            if new_install_parent not in installed_modules:
                new_installs.append(f"%pip install {new_install_parent}\n")
                installed_modules.append(new_install_parent)
        else:
            print(f"Reader/Tool not found: {module}\nKeeping line as is.")
            new_lines.append(line)

    return new_lines, new_installs, installed_modules


def parse_lines(
    lines: List[str], installed_modules: List[str], verbose: bool = False
) -> Tuple[List[str], List[str]]:
    with open(mappings_path) as f:
        mappings = json.load(f)

    new_installs = []
    new_lines = []
    just_found_imports = False
    skipped_lines = 0

    for idx, line in enumerate(lines):
        this_new_lines = []
        this_new_installs = []
        this_installed_modules = []

        if skipped_lines != 0:
            skipped_lines -= 1

        if just_found_imports and skipped_lines > 0:
            continue
        else:
            just_found_imports = False

        if (
            "from llama_index." in line
            or "from llama_index import" in line
            or "from llama_hub." in line
        ):
            (
                this_new_lines,
                this_new_installs,
                this_installed_modules,
                skipped_lines,
            ) = _parse_from_imports(
                mappings=mappings,
                installed_modules=installed_modules,
                line_idx=idx,
                lines=lines,
                verbose=verbose,
            )
            just_found_imports = True

        elif "download_loader(" in line or "download_tool(" in line:
            (
                this_new_lines,
                this_new_installs,
                this_installed_modules,
            ) = _parse_hub_downloads(
                mappings=mappings,
                installed_modules=installed_modules,
                line=line,
            )

        elif not just_found_imports:
            this_new_lines = [line]

        new_lines += this_new_lines
        new_installs += this_new_installs
        installed_modules += this_installed_modules
        installed_modules = list(set(installed_modules))

    return new_lines, list(set(new_installs))


def _cell_installs_llama_hub(cell) -> bool:
    lines = cell["source"]
    llama_hub_partial_statements = [
        "pip install llama-hub",
        "import download_loader",
        "import download_tool",
    ]

    if len(lines) > 1:
        return False
    if cell["cell_type"] == "code" and any(
        el in lines[0] for el in llama_hub_partial_statements
    ):
        return True
    return False


def _format_new_installs(new_installs):
    if new_installs:
        new_installs = list(set(new_installs))
        return new_installs[:-1] + [new_installs[-1].replace("\n", "")]
    return new_installs


def upgrade_nb_file(file_path):
    print(f"\n=====================\n{file_path}\n", flush=True)
    with open(file_path) as f:
        notebook = json.load(f)

    verbose = False
    if file_path == "../docs/examples/managed/manage_retrieval_benchmark.ipynb":
        verbose = True

    installed_modules = ["llama-index-core"]  # default installs
    cur_cells = []
    new_installs = []
    first_code_idx = -1
    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if verbose:
                print(f"cell: {cell}", flush=True)
            if first_code_idx == -1:
                first_code_idx = idx

            code = cell["source"]

            new_lines, cur_new_installs = parse_lines(code, installed_modules, verbose)
            new_installs += cur_new_installs

            cell["source"] = new_lines

        cur_cells.append(cell)

    if len(new_installs) > 0:
        notebook["cells"] = cur_cells
        new_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": _format_new_installs(new_installs),
        }
        cur_cells.insert(first_code_idx, new_cell)

    cur_cells = [cell for cell in cur_cells if not _cell_installs_llama_hub(cell)]
    notebook["cells"] = cur_cells
    with open(file_path, "w") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def upgrade_py_md_file(file_path: str) -> None:
    with open(file_path) as f:
        lines = f.readlines()

    installed_modules = ["llama-index-core"]  # default installs
    new_lines, new_installs = parse_lines(lines, installed_modules)

    with open(file_path, "w") as f:
        f.write("".join(new_lines))

    if len(new_installs) > 0:
        print("New installs:")
    for install in new_installs:
        print(install.strip().replace("%", ""))


def upgrade_file(file_path: str) -> None:
    if file_path.endswith(".ipynb"):
        upgrade_nb_file(file_path)
    elif file_path.endswith((".py", ".md")):
        upgrade_py_md_file(file_path)
    else:
        raise Exception(f"File type not supported: {file_path}")


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") and part not in [".", ".."] for part in path.parts)


def upgrade_dir(input_dir: str) -> None:
    file_refs = list(Path(input_dir).rglob("*.py"))
    file_refs += list(Path(input_dir).rglob("*.ipynb"))
    file_refs += list(Path(input_dir).rglob("*.md"))
    file_refs = [x for x in file_refs if not _is_hidden(x)]
    for file_ref in file_refs:
        if file_ref.is_file():
            upgrade_file(str(file_ref))
