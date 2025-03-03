import os
import subprocess
from pathlib import Path

path = os.getcwd() + '/ui_windows/ui_files'

for p in os.listdir(path):
    pa = path / Path(p)
    if (pa.suffix) == '.ui':
        subprocess.run(["pyside6-uic", pa, "-o", f"{os.getcwd()}/ui_windows/{pa.stem}.py"])