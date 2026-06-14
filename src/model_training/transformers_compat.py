def patch_transformers_hybrid_cache() -> None:
    """Map missing transformers.HybridCache to DynamicCache for newer peft."""
    try:
        import transformers
    except Exception:
        return

    if hasattr(transformers, "HybridCache"):
        return

    lazy_module_cls = type(transformers)
    original_getattr = getattr(lazy_module_cls, "__getattr__", None)
    if original_getattr is not None and not getattr(lazy_module_cls, "_echo_memory_hybrid_cache_patch", False):
        def _compat_getattr(self, name):
            if name == "HybridCache":
                return original_getattr(self, "DynamicCache")
            return original_getattr(self, name)

        lazy_module_cls.__getattr__ = _compat_getattr
        lazy_module_cls._echo_memory_hybrid_cache_patch = True

    try:
        transformers.HybridCache = transformers.DynamicCache
    except Exception:
        pass
