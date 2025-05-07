import pygame
import os

def test_controller():
    pygame.init()
    pygame.joystick.init()

    WINDOW_SIZE = (800, 600)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Controller Test")
    
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    if pygame.joystick.get_count() == 0:
        print("No controller detected!")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Controller: {joystick.get_name()}")
    
    font = pygame.font.Font(None, 36)
    
    running = True
    while running:
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        left_stick_x = joystick.get_axis(0)
        left_stick_y = joystick.get_axis(1)
        pygame.draw.circle(screen, WHITE, (200, 200), 50, 2) 
        stick_pos = (
            200 + left_stick_x * 40,
            200 + left_stick_y * 40
        )
        pygame.draw.circle(screen, RED, stick_pos, 10)  
        
        right_stick_x = joystick.get_axis(2)
        right_stick_y = joystick.get_axis(3)
        pygame.draw.circle(screen, WHITE, (400, 200), 50, 2)  
        stick_pos = (
            400 + right_stick_x * 40,
            200 + right_stick_y * 40
        )
        pygame.draw.circle(screen, RED, stick_pos, 10) 
        
        left_trigger = joystick.get_axis(4)
        right_trigger = joystick.get_axis(5)
        
        pygame.draw.rect(screen, WHITE, (100, 300, 20, 100), 2)
        trigger_height = int((left_trigger + 1) * 50)
        pygame.draw.rect(screen, GREEN, (100, 400 - trigger_height, 20, trigger_height))
        
        pygame.draw.rect(screen, WHITE, (500, 300, 20, 100), 2)
        trigger_height = int((right_trigger + 1) * 50)
        pygame.draw.rect(screen, GREEN, (500, 400 - trigger_height, 20, trigger_height))
        
        button_y = 450
        for i in range(joystick.get_numbuttons()):
            if i % 8 == 0:
                button_y += 40
            button_x = 100 + (i % 8) * 80
            
            if joystick.get_button(i):
                color = GREEN
            else:
                color = WHITE
            pygame.draw.circle(screen, color, (button_x, button_y), 15)
            text = font.render(str(i), True, BLACK if color == GREEN else WHITE)
            text_rect = text.get_rect(center=(button_x, button_y))
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    test_controller()