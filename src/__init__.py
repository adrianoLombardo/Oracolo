from pathlib import Path

# Expose the real package living under OcchioOnniveggente/src
_pkg_path = Path(__file__).resolve().parent.parent / 'OcchioOnniveggente' / 'src'
__path__ = [str(_pkg_path)]
