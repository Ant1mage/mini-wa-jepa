"""Isolate Mesa/EGL framebuffer limits without loading MetaDrive."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--software", action="store_true")
    p.add_argument("--output", type=Path, default=Path("artifacts/stage1/egl_probe.json"))
    args = p.parse_args()
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    if args.software:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    from panda3d.core import loadPrcFileData, FrameBufferProperties, Texture
    loadPrcFileData("", f"load-display p3headlessgl\negl-device-index {args.device}\n"
                    "window-type offscreen\naudio-library-name null\nwin-size 256 144\n"
                    "notify-level warning\nmultisamples 0")
    if args.software:
        # Panda3D 1.10 misclassifies Mesa llvmpipe when force_software is true.
        # LIBGL_ALWAYS_SOFTWARE and the reported renderer establish CPU use.
        loadPrcFileData("", "framebuffer-hardware false\nframebuffer-software false")
    from direct.showbase.ShowBase import ShowBase
    from direct.filter.FilterManager import FilterManager
    base = ShowBase()
    try:
        g = base.win.getGsg()
        result = dict(device_index=args.device, software_requested=args.software,
                      vendor=g.getDriverVendor(), renderer=g.getDriverRenderer(),
                      version=g.getDriverVersion(), pipe=base.pipe.getType().getName(), buffers=[])
        for samples in (0, 4, 8, 16):
            props = FrameBufferProperties()
            props.float_color = True
            props.setDepthBits(24)
            props.setMultisamples(samples)
            tex = Texture()
            tex.setFormat(Texture.F_rgba16)
            tex.setComponentType(Texture.T_float)
            manager = FilterManager(base.win, base.cam)
            quad = manager.renderSceneInto(colortex=tex, fbprops=props)
            result["buffers"].append(dict(msaa_requested=samples, success=quad is not None))
            manager.cleanup()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        base.destroy()


if __name__ == "__main__":
    main()
