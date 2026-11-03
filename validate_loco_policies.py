import scripts.validate_loco_policies as _module

for _name in dir(_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = [name for name in dir(_module) if not name.startswith("__")]
