
# --- IPython compatibility shim --------------------------------------------
def _ensure_get_ipython():
    try:
        from IPython import get_ipython as _real_get_ipython
        ip = _real_get_ipython()
        if ip is not None:
            return _real_get_ipython  # use real one
    except Exception:
        pass

    # Fallback: provide a minimal stub with .system() & .getoutput() like IPython
    import subprocess

    class _FakeIPython:
        def system(self, cmd):
            # mimic `!cmd` side effect (print output) and return exit code
            proc = subprocess.run(cmd, shell=True)
            return proc.returncode

        def getoutput(self, cmd):
            # like IPython's getoutput: return stdout as a list of lines
            out = subprocess.run(cmd, shell=True, text=True,
                                 capture_output=True).stdout
            return out.splitlines()

        # optional: emulate run_line_magic for a few cases if you used it
        def run_line_magic(self, name, line):
            raise RuntimeError(f"Magic %{name} not available outside IPython")

    def get_ipython():
        return _FakeIPython()

    return get_ipython

# expose get_ipython in the module namespace
get_ipython = _ensure_get_ipython()
# ---------------------------------------------------------------------------
