from io import StringIO
from typing import Any, Callable, List, Optional

import pandas as pd
from llama_index.core.node_parser.relational.base_element import (
    BaseElementNodeParser,
    Element,
)
from llama_index.core.schema import BaseNode, TextNode


def md_to_df(md_str: str) -> pd.DataFrame:
    """Convert Markdown to dataframe."""
    # Replace " by "" in md_str
    md_str = md_str.replace('"', '""')

    # Remove the second line (table header separator)
    lines = md_str.split("\n")

    # If the input does not follow markdown pipe table, bail out early (empty or not enough lines)
    if not lines or len(lines) < 2:
        return None

    # Remove the second line (table header separator)
    del lines[1]

    # Process all lines to replace markdown table pipes with expected CSV
    # - Assumption: each line starts and ends with pipes (|), so the valid items are between
    #   Remove first 2 and last 2 chars as done originally after replacement
    #   Also, do not rebuild md_str until after transformation

    processed_lines = []
    for line in lines:
        # Replace "|" with '","' efficiently, then cut first two and last two chars
        # As an additional safety for tables that may not have leading/trailing pipes,
        # fall back gracefully to empty string if line too short
        csv_line = line.replace("|", '","')
        if len(csv_line) >= 4:
            csv_line = csv_line[2:-2]
        else:
            csv_line = ''
        processed_lines.append(csv_line)

    md_str = "\n".join(processed_lines)

    # Check if the table is empty
    if not md_str:
        return None

    # Avoid pandas parsing being a bottleneck by disabling automatic dtype inference
    # - For markdown document tables, all fields are strings, so force string dtype
    # - Also, specify quoting only as needed (already double quoted in conversion)
    try:
        # Quoting minimal, separator default
        # For speed, let pandas guess column names; do not pass extra arguments
        return pd.read_csv(StringIO(md_str), dtype=str, engine='c', quoting=pd.io.common.csv.QUOTE_MINIMAL)
    except Exception:
        # In case of unexpected parse errors, return None as before
        return None


class MarkdownElementNodeParser(BaseElementNodeParser):
    """Markdown element node parser.

    Splits a markdown document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """

    @classmethod
    def class_name(cls) -> str:
        return "MarkdownElementNodeParser"

    def get_nodes_from_node(self, node: TextNode) -> List[BaseNode]:
        """Get nodes from node."""
        elements = self.extract_elements(
            node.get_content(),
            table_filters=[self.filter_table],
            node_id=node.id_,
        )
        table_elements = self.get_table_elements(elements)
        # extract summaries over table elements
        self.extract_table_summaries(table_elements)
        # convert into nodes
        # will return a list of Nodes and Index Nodes
        return self.get_nodes_from_elements(elements)

    def extract_elements(
        self,
        text: str,
        node_id: Optional[str] = None,
        table_filters: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> List[Element]:
        # get node id for each node so that we can avoid using the same id for different nodes
        """Extract elements from text."""
        lines = text.split("\n")
        currentElement = None

        elements: List[Element] = []
        # Then parse the lines
        for line in lines:
            if line.startswith("```"):
                # check if this is the end of a code block
                if currentElement is not None and currentElement.type == "code":
                    elements.append(currentElement)
                    currentElement = None
                    # if there is some text after the ``` create a text element with it
                    if len(line) > 3:
                        elements.append(
                            Element(
                                id=f"id_{len(elements)}",
                                type="text",
                                element=line.lstrip("```"),
                            )
                        )

                elif line.count("```") == 2 and line[-3] != "`":
                    # check if inline code block (aka have a second ``` in line but not at the end)
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}",
                        type="code",
                        element=line.lstrip("```"),
                    )
                elif currentElement is not None and currentElement.type == "text":
                    currentElement.element += "\n" + line
                else:
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )

            elif currentElement is not None and currentElement.type == "code":
                currentElement.element += "\n" + line

            elif line.startswith("|"):
                if currentElement is not None and currentElement.type != "table":
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
            elif line.startswith("#"):
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}",
                    type="title",
                    element=line.lstrip("#"),
                    title_level=len(line) - len(line.lstrip("#")),
                )
            else:
                if currentElement is not None and currentElement.type != "text":
                    elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
        if currentElement is not None:
            elements.append(currentElement)

        for idx, element in enumerate(elements):
            if element.type == "table":
                should_keep = True
                perfect_table = True

                # verify that the table (markdown) have the same number of columns on each rows
                table_lines = element.element.split("\n")
                table_columns = [len(line.split("|")) for line in table_lines]
                if len(set(table_columns)) > 1:
                    # if the table have different number of columns on each rows, it's not a perfect table
                    # we will store the raw text for such tables instead of converting them to a dataframe
                    perfect_table = False

                # verify that the table (markdown) have at least 2 rows
                if len(table_lines) < 2:
                    should_keep = False

                # apply the table filter, now only filter empty tables
                if should_keep and perfect_table and table_filters is not None:
                    should_keep = all(tf(element) for tf in table_filters)

                # if the element is a table, convert it to a dataframe
                if should_keep:
                    if perfect_table:
                        table = md_to_df(element.element)

                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table",
                            element=element,
                            table=table,
                        )
                    else:
                        # for non-perfect tables, we will store the raw text
                        # and give it a different type to differentiate it from perfect tables
                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table_text",
                            element=element.element,
                            # table=table
                        )
                else:
                    elements[idx] = Element(
                        id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                        type="text",
                        element=element.element,
                    )
            else:
                # if the element is not a table, keep it as to text
                elements[idx] = Element(
                    id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                    type="text",
                    element=element.element,
                )

        # merge consecutive text elements together for now
        merged_elements: List[Element] = []
        for element in elements:
            if (
                len(merged_elements) > 0
                and element.type == "text"
                and merged_elements[-1].type == "text"
            ):
                merged_elements[-1].element += "\n" + element.element
            else:
                merged_elements.append(element)
        elements = merged_elements
        return merged_elements

    def filter_table(self, table_element: Any) -> bool:
        """Filter tables."""
        table_df = md_to_df(table_element.element)

        # check if table_df is not None, has more than one row, and more than one column
        return table_df is not None and not table_df.empty and len(table_df.columns) > 1
