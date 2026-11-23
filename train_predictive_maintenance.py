import scripts.train_ims_model as _module

for _name in dir(_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = [name for name in dir(_module) if not name.startswith("__")]

if __name__ == "__main__":
    _module.main()
