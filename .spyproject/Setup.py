import subprocess
import pygame
import numpy as np
import gym
from gym import spaces
import files
import threading

from builtins import exec


from scipy import linalg
import font

image_fundo = pygame.image.load('assets/fundo3.jpg')
image_car = pygame.image.load('assets/carro.png')


class InvertedPendulum():
    pygame.init()
    # Initialize environment
    def __init__(self, xRef=np.random.uniform(-0.9, 0.9), randomParameters=False, randomSensor=False, randomActuator=False):
        # System parameters.
        self.tau = 0.01
        if not randomParameters:
            self.g = 9.8
            self.M = 1.0
            self.m = 0.1
            self.l = 0.5
        else:
            self.g = 9.8 + 0.098*np.random.randn()
            self.M = 1.0 + 0.1 * np.random.randn()
            self.m = 0.1 + 0.01*np.random.randn()
            self.l = 0.5 + 0.05*np.random.randn()

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        # posição SETPOINT
        self.xRef = xRef

        # Caixas de Texto
        self.textBox_width = 120
        self.textBox_height = 25

        def TextBox(x, y):
            return pygame.Rect(x, y, self.textBox_width, self.textBox_height)
        self.activebox_p = False
        self.activebox_i = False
        self.activebox_d = False
        self.base_font = pygame.font.Font(None, 25)
        self.user_text_p = '0'
        self.user_text_i = '0'
        self.user_text_d = '0'

        self.input_rect_p = TextBox(680, 10)
        self.input_rect_i = TextBox(680, 10*2+self.textBox_height)
        self.input_rect_d = TextBox(680, 10*3 + self.textBox_height*2)
        self.color_active = pygame.Color('Red')
        self.color_passive = pygame.Color('White')
        self.color = self.color_passive

        # Informações carro
        self.cartWidth = 160
        self.cartHeight = 80

        # pendulo
        self.pendulumLength = 200
        # linha de base
        self.baseLine = 350
        # tela
        self.screenWidth = 800
        self.screenHeight = 400

        # Variable to see if simulation ended.
        self.finish = False

        # Variable to see if there is randomness in the sensors and actuators.
        self.randomSensor = randomSensor
        self.randomActuator = randomActuator

        # Create a random observation.
        self.reset()

        # Create screen.
        self.screen = pygame.display.set_mode(
            (self.screenWidth, self.screenHeight))
        self.screen.fill('White')
        pygame.display.set_caption('Pêndulo Invertido')  # nome do display

        # Create a clock object.
        self.clock = pygame.time.Clock()
        pygame.display.update()

    # Close environment window.

    def close(self):
        pygame.quit()

    # Reset system with a new random initial position.
    def reset(self):
        self.observation = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()

    # Insert noise on the sensors.
    def noise_sensors(self, observation, noiseVar=0.01):
        observation[0] = observation[0] + noiseVar*np.random.randn()
        observation[1] = observation[1] + noiseVar*np.random.randn()
        observation[2] = observation[2] + noiseVar*np.random.randn()
        observation[3] = observation[3] + noiseVar*np.random.randn()
        return observation

    # Insert noise on actuator.
    def noise_actuator(self, action, noiseVar=0.01):
        action += noiseVar * np.random.randn()
        return action

    # Display object.
    def render(self):
        # resetar carro
        def restart():
            self.observation[0] = 0
            self.observation[1] = 0
            self.observation[2] = 0
            self.observation[3] = 0
        # Check for all possible types of player input.
        for event in pygame.event.get():
            # Command for closing the window.
            if (event.type == pygame.QUIT):
                pygame.quit()
                self.finish = True
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.input_rect_p.collidepoint(event.pos):
                    self.activebox_p = True
                else:
                    self.activebox_p = False

                if self.input_rect_i.collidepoint(event.pos):
                    self.activebox_i = True
                else:
                    self.activebox_i = False

                if self.input_rect_d.collidepoint(event.pos):
                    self.activebox_d = True
                else:
                    self.activebox_d = False

            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_LEFT):
                    self.xRef -= 0.01

                elif (event.key == pygame.K_RIGHT):
                    self.xRef += 0.01

                elif (event.key == pygame.K_SPACE):
                    self.step(200*np.random.randn())

                elif (event.key == pygame.K_r):
                    restart()
                # Check for backspace
                if event.key == pygame.K_BACKSPACE and self.activebox_p == True:
                    self.user_text_p = self.user_text_p[:-1]
                elif self.activebox_p == True:
                    self.user_text_p += event.unicode

                if event.key == pygame.K_BACKSPACE and self.activebox_i == True:
                    self.user_text_i = self.user_text_i[:-1]
                elif self.activebox_i == True:
                    self.user_text_i += event.unicode

                if event.key == pygame.K_BACKSPACE and self.activebox_d == True:
                    self.user_text_d = self.user_text_d[:-1]
                elif self.activebox_d == True:
                    self.user_text_d += event.unicode

        # Apply surface over display.
        self.screen.fill('White')
        pygame.draw.line(self.screen, 'Black', (0, self.baseLine),
                         (self.screenWidth, self.baseLine))

        # Get position for cart.
        xCenter = self.screenHeight + self.screenHeight * self.observation[0]
        if (xCenter < -400 or xCenter > 1200):
            restart()

        # Get position for pendulum.
        pendX = xCenter + self.pendulumLength * np.sin(self.observation[2])
        pendY = self.baseLine - self.pendulumLength * \
            np.cos(self.observation[2])

        if not self.activebox_p and not self.activebox_i and not self.activebox_d:
            if self.user_text_p == "":
                self.user_text_p = "0"
            if self.user_text_i == "":
                self.user_text_i = "0"
            if self.user_text_d == "":
                self.user_text_d = "0"

        # Display objects.
        self.screen.blit(image_fundo, (0, 0))
        pygame.draw.line(self.screen,   "Yellow", (int(self.screenHeight + self.xRef * self.screenHeight), 0),
                         (int(self.screenHeight + self.xRef * self.screenHeight), self.baseLine), width=2)
        self.screen.blit(image_car, (xCenter - self.cartWidth /
                         2, self.baseLine - self.cartHeight/2))
        pygame.draw.line(self.screen,   (101, 101, 101),
                         (xCenter, self.baseLine), (pendX, pendY), width=6)
        pygame.draw.circle(self.screen, (101, 101, 101),
                           (xCenter, self.baseLine), 10)

        if self.activebox_p:
            color_p = self.color_active
        else:
            color_p = self.color_passive

        if self.activebox_i:
            color_i = self.color_active
        else:
            color_i = self.color_passive

        if self.activebox_d:
            color_d = self.color_active
        else:
            color_d = self.color_passive

        pygame.draw.rect(self.screen, color_p, self.input_rect_p, 2)
        pygame.draw.rect(self.screen, color_i, self.input_rect_i, 2)
        pygame.draw.rect(self.screen, color_d, self.input_rect_d, 2)

        text_surface_p = self.base_font.render(
            self.user_text_p, True, (0, 0, 0))
        text_surface_i = self.base_font.render(
            self.user_text_i, True, (0, 0, 0))
        text_surface_d = self.base_font.render(
            self.user_text_d, True, (0, 0, 0))

        text_p = self.base_font.render(
            "P", True, (0, 0, 0))
        text_i = self.base_font.render(
            "I", True, (0, 0, 0))
        text_d = self.base_font.render(
            "D", True, (0, 0, 0))

        self.screen.blit(
            text_surface_p, (self.input_rect_p.x+5, self.input_rect_p.y+5))

        self.screen.blit(
            text_p, (self.input_rect_p.x-20, self.input_rect_p.y+5))

        self.screen.blit(
            text_surface_i, (self.input_rect_i.x+5, self.input_rect_i.y+5))

        self.screen.blit(
            text_i, (self.input_rect_i.x-20, self.input_rect_i.y+5))

        self.screen.blit(
            text_surface_d, (self.input_rect_d.x+5, self.input_rect_d.y+5))

        self.screen.blit(
            text_d, (self.input_rect_d.x-20, self.input_rect_d.y+5))

        self.input_rect_p.w = max(80, text_surface_p.get_width()+10)
        self.input_rect_i.w = max(80, text_surface_i.get_width()+10)
        self.input_rect_d.w = max(80, text_surface_d.get_width()+10)

        pygame.display.update()
        # Limit framerate.
        self.clock.tick(60)

    # Perform a step.
    def step(self, force):
        if self.randomActuator:
            force = self.noise_actuator(force)
        x1 = self.observation[0]
        x2 = self.observation[1]
        x3 = self.observation[2]
        x4 = self.observation[3]
        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (force + self.m * self.l * x4**2 * np.sin(x3)) /
                 (self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)**2 / (self.M + self.m)))
        x2dot = (force + self.m * self.l * x4**2 * np.sin(x3))/(self.M +
                                                                self.m) - self.m * self.l * x4dot * np.cos(x3) / (self.M + self.m)
        self.observation[0] = x1 + self.tau * x2
        self.observation[1] = x2 + self.tau * x2dot
        self.observation[2] = x3 + self.tau * x4
        self.observation[3] = x4 + self.tau * x4dot
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()

