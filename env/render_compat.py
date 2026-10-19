"""Explicit Panda3D framebuffer adapter for MetaDrive's hardcoded 16x MSAA.

Only anti-aliasing requests are capped. Physics, maps and cameras remain native
MetaDrive. No third-party source files are modified. Call restore on exit.
"""
def limit_filter_msaa(limit: int):
    from direct.filter.FilterManager import FilterManager
    from panda3d.core import FrameBufferProperties

    if limit not in (0, 2, 4, 8):
        raise ValueError("Supported caps: 0, 2, 4, 8")
    original = FilterManager.createBuffer
    original_alias = FilterManager.create_buffer
    changes = []

    def create_buffer(self, name, xsize, ysize, texgroup, depthbits=True, fbprops=None):
        if fbprops is not None and fbprops.getMultisamples() > limit:
            changes.append(dict(buffer=name, requested=fbprops.getMultisamples(), actual_request=limit))
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setMultisamples(limit)
        return original(self, name, xsize, ysize, texgroup, depthbits, fbprops)

    FilterManager.createBuffer = create_buffer
    FilterManager.create_buffer = create_buffer

    def restore():
        FilterManager.createBuffer = original
        FilterManager.create_buffer = original_alias

    return changes, restore
