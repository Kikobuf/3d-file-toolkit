"""Entry point for the 3dft command."""
import os, sys

def main():
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
    sys.path.insert(0, scripts_dir)
    # Delegate to the 3dft dispatcher
    dispatcher = os.path.join(os.path.dirname(os.path.dirname(__file__)), "3dft")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_3dft", dispatcher)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == "__main__":
    main()
