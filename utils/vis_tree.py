from graphviz import Digraph
import json
from .check_ans import is_equiv

def safe_label(text):
    return json.dumps(text, ensure_ascii=False)

def visualize_tree(node_list, output_path="tree", view=False, gt:str=""):
    """
    Visualize a tree from a list of dicts using Graphviz (left-to-right layout).
    
    Args:
        node_list: List of dicts, each with keys: id, parent_id, action
        output_path: Output file name (without extension)
        view: Whether to open the rendered file automatically
    """
    dot = Digraph(format="png")

    dot.attr(rankdir="LR")

    dot.attr("node", shape="rectangle", style="filled", fillcolor="lightblue", fontname="Helvetica")

    for node in node_list:
        node_id = str(node["id"])
        node_create_iter = node.get("create_iter", node.get("create_step", 0))
        label = node["action"]
        if isinstance(label, dict):
            label = node["action"]["content"] if 'content' in node["action"] else "None"
        label = safe_label(label)
        label = f"node_id:{node_id}\\ncreate_at_step:{node_create_iter}\\n{label}"
        
        if node.get("is_terminal", False):
            if is_equiv(node["action"].get("content", ""), gt):
                fillcolor = "lightgreen"
            else:
                fillcolor = "lightyellow"
        else:
            fillcolor = "lightblue"
        
        dot.node(node_id, label, style="filled", fillcolor=fillcolor)

    for node in node_list:
        if node["parent_id"] is not None:
            parent_id = str(node["parent_id"])
            child_id = str(node["id"])
            dot.edge(parent_id, child_id)

    dot.render(output_path, view=view)
