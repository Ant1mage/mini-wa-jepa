"""Shared visualization API; the SDK is loaded only when visualization is used."""

__all__ = ["RerunLogger", "colorize_bev_4ch"]


def __getattr__(name):
    if name in __all__:
        from . import rerun_logger
        return getattr(rerun_logger, name)
    raise AttributeError(name)
