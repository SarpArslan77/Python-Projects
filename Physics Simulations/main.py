import pygame
import numpy as np

from light import Light

# Initialize the pygame.
pygame.init()

# Screen dimensions
screen_width: int = 800
screen_height: int = 600

# Create the screen.
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Light Intereference Simulation")

# Colors
colors: dict = {
    "dark_blue" : (15, 5, 100),
    "white" : (255, 255, 255),
    "black" : (0, 0, 0),
    "yellow" : (255, 255, 0),
    "red" : (255, 0, 0),
    "green" : (0, 255, 0),
    "blue" : (0, 0, 255),
}

# Create the light sources.
source1 = Light(50, 150, 5, 100)
source2 = Light(50, 450, 5, 100)

def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return np.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))

# Game loop
running: bool = True
while running:
    # User inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # Drawing

    # Background 
    screen.fill(colors["white"])
    # Wall
    #pygame.draw.rect(screen, colors["black"], (700, 50, 50, 500), width=1)
    # Light sources
    pygame.draw.circle(screen, colors["red"], (source1.position_x, source1.position_y), 5)
    pygame.draw.circle(screen, colors["red"], (source2.position_x, source2.position_y), 5)

    # Check every pixel on the grid to see intereference of light sources.
    #   It is being accepted that, light sources can not emit light in the back half.
    for y in range(screen_height):
        for x in range(source1.position_x, screen_width):
            # Assume the light sources are no-zone.
            if (not(150 < y < 155) and not(450 < y < 455)) and \
                (not(50 < x < 55)):
                # Calculate the distances to the sources.
                distance_to_source1 = calculate_distance(x, y, source1.position_x, source1.position_y)
                distance_to_source2 = calculate_distance(x, y, source2.position_x, source2.position_y)
                # Calculate the values for waves.
                source1_val: float = source1(distance_to_source1)
                source2_val: float = source2(distance_to_source2)
                
                # Calculate the intensity
                total_value: float = source1_val + source2_val
                opacity: float = total_value / (source1.amplitude + source2.amplitude)
                color = (255, 0, 0, opacity) # Tone of red

                # Draw the results.
                pygame.draw.line(screen, color, (x, y), (x, y))

    # Update the display.
    pygame.display.flip()

# Quit pygame
pygame.quit()
    