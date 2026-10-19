"""Keyboard control in a separate process; this module imports no graphics backend."""
import multiprocessing as mp
from queue import Empty
import time

import numpy as np


def _keyboard_worker(queue,stop,headless,test_keys):
    import os
    if headless:os.environ['SDL_VIDEODRIVER']='dummy'
    import pygame
    pygame.display.init();pygame.font.init()
    screen=pygame.display.set_mode((600,220));pygame.display.set_caption('mini-wa-jepa | isolated controls')
    font=pygame.font.Font(None,24);clock=pygame.time.Clock()
    # Expose the actual SDL client window; compositor decorations may share its title.
    queue.put(('ready',dict(driver=pygame.display.get_driver(),
                           window_id=pygame.display.get_wm_info().get('window'))))
    for name in test_keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.key.key_code(name)))
    try:
        while not stop.is_set():
            for event in pygame.event.get():
                if event.type==pygame.QUIT:queue.put(('key','q'));stop.set()
                elif event.type in (pygame.KEYDOWN,pygame.KEYUP):
                    name=pygame.key.name(event.key)
                    queue.put(('down' if event.type==pygame.KEYDOWN else 'up',name))
            screen.fill((20,24,32))
            for i,text in enumerate(['Controls affect the simulator; Rerun is the viewer.',
                'SPACE pause / resume     N single simulation step',
                'R reset     Q quit     M manual / expert',
                'W / S throttle / brake     A / D steering']):
                screen.blit(font.render(text,True,(225,230,240)),(20,25+i*42))
            pygame.display.flip();clock.tick(30)
    finally:pygame.display.quit();pygame.font.quit()


class KeyboardControls:
    def __init__(self,headless=False,test_keys=()):
        context=mp.get_context('spawn')
        self.queue=context.Queue();self.stop=context.Event()
        self.process=context.Process(target=_keyboard_worker,args=(self.queue,self.stop,headless,test_keys),daemon=True)
        self.paused=self.single_step=self.reset_requested=self.manual=self.quit=False
        self.held=set();self.initialized=False;self.desktop_info={}
        self.process.start()
        deadline=time.monotonic()+10
        while not self.initialized and time.monotonic()<deadline:
            self.poll();time.sleep(.02)
        if not self.initialized:
            self.close();raise RuntimeError('Isolated keyboard process failed to start')

    def poll(self):
        while True:
            try:kind,name=self.queue.get_nowait()
            except Empty:break
            if kind=='ready':self.initialized=True;self.desktop_info=name or {}
            elif kind=='up':self.held.discard(name)
            else:
                self.held.add(name)
                if name=='space':self.paused=not self.paused
                elif name=='n':self.single_step=True;self.paused=True
                elif name=='r':self.reset_requested=True
                elif name=='m':self.manual=not self.manual
                elif name=='q':self.quit=True
        if self.initialized and not self.process.is_alive():self.quit=True

    def ready(self):
        self.poll()
        if self.single_step:self.single_step=False;return True
        return not self.paused

    def wait_until_ready(self):
        while not self.ready() and not self.quit and not self.reset_requested:time.sleep(.02)

    def manual_action(self):
        return np.array([float('a' in self.held)-float('d' in self.held),
                         float('w' in self.held)-float('s' in self.held)],np.float32)

    def close(self):
        self.stop.set();self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=3)
        self.queue.close()
