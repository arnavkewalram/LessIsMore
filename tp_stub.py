"""
Fallback stub so that `import tensor_parallel as tp` never crashes when the
real package is absent (we don’t use it in LoRA fine-tuning).
"""
class _TPStub:                       # provides dummy attrs that raise on use
    def __getattr__(self, name):
        raise ImportError(
            "tensor_parallel is not installed and this part of the code "
            "tried to access tp.%s ; either install the real package "
            "(`pip install tensor_parallel`) or remove that call." % name)

tp = _TPStub()
