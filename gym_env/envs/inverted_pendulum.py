import pygame
import numpy as np
import gymnasium  as gym
from gymnasium import spaces
import gym_env

image_fundo = pygame.image.load('./gym_env/assets/fundo3.jpg')
image_car = pygame.image.load('./gym_env/assets/carro.png')
icon_logo = pygame.image.load('./gym_env/assets/logo_ifes.png')

class InvertedPendulum(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    pygame.display.set_icon(icon_logo)

    def __init__(self, xRef=np.random.uniform(-0.9, 0.9), render_mode=None, randomParameters=False, randomSensor=False, randomActuator=False):
        # screen
        pygame.display.set_caption('Pêndulo Invertido')  
        heigh = pygame.display.Info().current_h -60
        self.window_size_width = pygame.display.Info().current_w
        self.window_size_height = heigh

        self.theta_threshold_radians = 60 * np.pi / 180 
        self.x_threshold = 1
        high = np.array([self.x_threshold*2, np.finfo(np.float32).max, self.theta_threshold_radians*2, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-high, high, dtype = np.float32),
                "target": spaces.Box(-self.x_threshold, self.x_threshold, dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(50)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = pygame.time.Clock()

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

        # setpoint
        self.xRef = xRef

        # car
        self.cartWidth = 160
        self.cartHeight = 80

        # pendulum
        self.pendulumLength = 200

        # line
        self.baseLine = heigh/2

        # TextField
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
        self.input_rect_i = TextBox(680, 10*2 + self.textBox_height)
        self.input_rect_d = TextBox(680, 10*3 + self.textBox_height*2)
        self.color_active = pygame.Color('Red')
        self.color_passive = pygame.Color('Black')
        self.color = self.color_passive

        # Variable to see if simulation ended.
        self.finish = False

        # Variable to see if there is randomness in the sensors and actuators.
        self.randomSensor = randomSensor
        self.randomActuator = randomActuator

        # Create a random observation.
        self.reset()

    def _get_obs(self):
        if self.randomSensor:
            return {"agent": self.noise_sensors(self._agent_location.copy()), "target": self._target_location}
        else:
            return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": (self._agent_location[0] - self._target_location)
        }

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self._target_location = self.xRef

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation,info

    def step(self, action):
        self._target_location = self.xRef
        if self.randomActuator:
            action = self.noise_actuator(action)
        x1 = self._agent_location[0]
        x2 = self._agent_location[1]
        x3 = self._agent_location[2]
        x4 = self._agent_location[3]
        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (action + self.m * self.l * x4**2 * np.sin(x3)) /
                 (self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)**2 / (self.M + self.m)))
        x2dot = (action + self.m * self.l * x4**2 * np.sin(x3))/(self.M +
                                                                self.m) - self.m * self.l * x4dot * np.cos(x3) / (self.M + self.m)
        self._agent_location[0] = x1 + self.tau * x2
        self._agent_location[1] = x2 + self.tau * x2dot
        self._agent_location[2] = x3 + self.tau * x4
        self._agent_location[3] = x4 + self.tau * x4dot

        # We use `np.clip` to make sure we don't leave the grid
        terminated = np.array_equal(self._agent_location[0], self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_width, self.window_size_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        # resetar carro
        def restart():
            self._agent_location[0] = 0
            self._agent_location[1] = 0
            self._agent_location[2] = 0
            self._agent_location[3] = 0
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

        canvas = pygame.Surface((self.window_size_width, self.window_size_height))
        canvas.fill((255, 255, 255))

        pygame.draw.line(canvas, 'Black', (0, self.baseLine),
                         (self.window_size_width, self.baseLine))

        # Get position for cart.
        xCenter = self.window_size_height + self.window_size_height * self._agent_location[0]
        if (xCenter < -2000 or xCenter > 2000):
            restart()

        # Get position for pendulum.
        pendX = xCenter + self.pendulumLength * np.sin(self._agent_location[2])
        pendY = self.baseLine - self.pendulumLength * \
            np.cos(self._agent_location[2])

        if not self.activebox_p and not self.activebox_i and not self.activebox_d:
            if self.user_text_p == "":
                self.user_text_p = "0"
            if self.user_text_i == "":
                self.user_text_i = "0"
            if self.user_text_d == "":
                self.user_text_d = "0"

        # Display objects.
        #self.screen.blit(image_fundo, (0, 0))
        pygame.draw.line(canvas,   "Yellow", (int(self.window_size_height + self.xRef * self.window_size_height), 0),
                         (int(self.window_size_height + self.xRef * self.window_size_height), self.baseLine), width=2)
        canvas.blit(image_car, (xCenter - self.cartWidth /
                         2, self.baseLine - self.cartHeight/2))
        pygame.draw.line(canvas,   (101, 101, 101),
                         (xCenter, self.baseLine), (pendX, pendY), width=6)
        pygame.draw.circle(canvas, (101, 101, 101),
                           (xCenter, self.baseLine), 10)    

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

