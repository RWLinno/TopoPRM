"""Apply the executed TRL availability shim inside an explicit virtualenv only."""
from pathlib import Path
import argparse,hashlib,importlib.metadata as md,importlib.util,sys
ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
assert md.version('trl')=='0.28.0' and md.version('transformers')=='5.5.4'
spec=importlib.util.find_spec('trl');p=Path(next(iter(spec.submodule_search_locations)))/'import_utils.py'
old=p.read_text();needle='from transformers.utils.import_utils import _is_package_available\n'
replacement='from transformers.utils.import_utils import _is_package_available as _hf_package_available\n\ndef _is_package_available(pkg_name, return_version=False):\n    value = _hf_package_available(pkg_name, return_version=return_version)\n    return value if return_version or not isinstance(value, tuple) else value[0]\n'
new=old.replace(needle,replacement)
assert needle in old or replacement in old,'Unexpected TRL implementation'
print('Shim target:',p);print('Patched SHA256:',hashlib.sha256(new.encode()).hexdigest())
if args.apply:
    assert sys.prefix!=sys.base_prefix,'Activate a dedicated virtualenv first'
    assert p.resolve().is_relative_to(Path(sys.prefix).resolve()),'Refuse to modify a shared installation'
    if new!=old:p.write_text(new)
    print('Availability shim ready; model and loss code unchanged.')
