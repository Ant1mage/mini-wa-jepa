"""Compare same-title decoration targeting with the actual SDL window ID."""
import ctypes as C
import os
import time
os.environ['SDL_VIDEODRIVER']='x11'
import pygame
from scripts.probe_keyboard_desktop import DesktopKeys
pygame.display.init()
pygame.display.set_mode((600,220))
pygame.display.set_caption('mini-wa-jepa | isolated controls')
time.sleep(.5)
pygame.event.get()
actual=pygame.display.get_wm_info()['window']
keys=DesktopKeys(actual)
x=keys.x
x.XQueryTree.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_ulong),C.POINTER(C.c_ulong),C.POINTER(C.POINTER(C.c_ulong)),C.POINTER(C.c_uint)]
x.XFetchName.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_char_p)]
x.XFree.argtypes=[C.c_void_p]
root=x.XDefaultRootWindow(keys.display)
a,b,children,n=C.c_ulong(),C.c_ulong(),C.POINTER(C.c_ulong)(),C.c_uint()
x.XQueryTree(keys.display,root,C.byref(a),C.byref(b),C.byref(children),C.byref(n))
wrong=None
for window in children[:n.value]:
 name=C.c_char_p()
 if x.XFetchName(keys.display,window,C.byref(name)):
  title=name.value.decode(errors='replace');x.XFree(C.cast(name,C.c_void_p))
  if title=='mini-wa-jepa | isolated controls' and window!=actual:wrong=window
x.XFree(children)
try:
 if wrong:
  keys.window=wrong;keys.tap('space');time.sleep(.1)
  print('decoration',wrong,'key_events',[str(e) for e in pygame.event.get() if e.type in (pygame.KEYDOWN,pygame.KEYUP)])
 keys.window=actual;keys.tap('space');time.sleep(.1)
 events=[e for e in pygame.event.get() if e.type in (pygame.KEYDOWN,pygame.KEYUP)]
 print('SDL client',actual,'key_events',[str(e) for e in events])
 assert any(e.type==pygame.KEYDOWN and e.key==pygame.K_SPACE for e in events)
finally:keys.close();pygame.quit()
