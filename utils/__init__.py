from .check_ans import is_equiv
from .misc import *

# Lazily expose visualize_tree; avoid importing graphviz when not needed
try:
	from .vis_tree import visualize_tree
except Exception as _e:
	def visualize_tree(*args, **kwargs):  # type: ignore
		raise ImportError(
			"visualize_tree requires graphviz; please install `graphviz` system package and the Python package `graphviz`.\n"
			f"Original error: {_e}"
		)
