"""INPUT: window_methods.tex editorial template from Results_keep.rtf.
OUTPUT: Table_Upload/Table_Window_Methods.tex only.
This table describes the window analysis; it contains no numerical estimates.
"""

# Resolve package imports when running this file directly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__ = "code.code_manuscript"

from pathlib import Path
from ..shared.io_utils import TABLE


def generate(ctx):
    TABLE.mkdir(exist_ok=True)
    (TABLE/'Table_Window_Methods.tex').write_text((Path(__file__).parent.parent/'shared/window_methods.tex').read_text())


if __name__ == "__main__":
    print("Running table_window_methods.py", flush=True)
    generate(None)
    print("Completed table_window_methods.py", flush=True)
