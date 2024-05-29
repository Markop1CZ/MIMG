import pygame
import os
import time
from PIL import Image
from img import MImg

img_dir = "test-images-output"
img_ext = ".buf"

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()

font = pygame.font.SysFont("Calibri", 20, True)

images = []
for f in os.listdir(img_dir):
    if os.path.splitext(f)[1] == img_ext:
        images.append(os.path.join(img_dir, f))

idx = 0

running = True
load = True
img_surf = None
while running:
    if load:
        t = time.time()
        img = MImg.from_file(images[idx])._image
        t2 = time.time()
        img_surf = pygame.image.fromstring(img.tobytes(), img.size, img.mode).convert()
        t3 = time.time()
        load = False

        print("loading={0:.02f}s pygame={1:.02f}s total={2:.02f}s".format(t2-t, t3-t2, t3-t))

    screen.fill((255, 255, 255))

    screen.blit(img_surf, (10, 10))
    screen.blit(font.render("{0:.0f}".format(clock.get_fps()), True, (0, 0, 0)), (0, 0))

    pygame.display.flip()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            running = False

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RIGHT:
                idx += 1
                load = True
            if e.key == pygame.K_LEFT:
                idx -= 1
                load = True
            
            if idx <= 0:
                idx = len(images)-1
            if idx >= len(images):
                idx = 0
    
    clock.tick(60)