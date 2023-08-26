import pygame
import numpy as np
from scipy.signal import place_poles
import gymnasium  as gym
from gymnasium import spaces

icon_logo = pygame.image.load('./gym_env/assets/logo_ifes.png')

class Elevator(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    pygame.init()
    pygame.display.set_caption('Elevator System')
    pygame.display.set_icon(icon_logo)
    # Initialize environment.
    def __init__(self, xRef=0.0, render_mode=None, randomParameters=False, randomSensor=False, randomActuator=False):
        # attributes
        self.x_threshold = 1
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-self.x_threshold, self.x_threshold, dtype = np.float32),
                "target": spaces.Box(-self.x_threshold, self.x_threshold, dtype = np.float32),
            }
        )
        self.action_space = spaces.Discrete(50)
       
        # System parameters.
        self.tau = 0.01
        self.g = 9.8
        self.M = 500.0
        self.m = 100.0
        self.b = 0.2
        # If there is randomness in the system parameters.
        if randomParameters:
            self.g += self.g/100 * np.random.randn()
            self.M += self.M/100 * np.random.randn()
            self.m += self.m/100 * np.random.randn()
            self.b += self.b/100 * np.random.randn()
        self.xRef = xRef

        # Drawing parameters.
        self.elevatorWidth = 80
        self.elevatorHeight = 100
        self.counterWeightWidth = 20
        self.counterWeightHeight = 40
        self.wheelRadius = 100
        self.baseLine = 100
        self.window_size_width = pygame.display.Info().current_w
        self.window_size_height = pygame.display.Info().current_h -60
        # Variable to see if simulation ended.
        self.finish = False
        
        # Variable to see if there is randomness in the sensors and actuators.
        self.randomSensor   = randomSensor
        self.randomActuator = randomActuator
        
        # Create a random observation.
        self.reset()
        
        # Create a clock object.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = pygame.time.Clock()
    
    def _get_obs(self):
        if self.randomSensor:
            return {"agent": self.noise_sensors(self._agent_location.copy()), "target": self._target_location}
        else:
            return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": (self._agent_location[0] - self._target_location)
        }
    
    # Reset system with a new random initial position.
    def reset(self,seed = None):
        super().reset(seed=seed)

        self._agent_location = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self._target_location = self.xRef

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation,info
    
    # Insert noise on the sensors.
    def noise_sensors(self, observation, noiseVar = 0.01):
        observation[0] = observation[0] + noiseVar*np.random.randn()
        observation[1] = observation[1] + noiseVar*np.random.randn()
        return observation
    
    # Insert noise on actuator.
    def noise_actuator(self, action, noiseVar = 0.01):
        action += noiseVar * np.random.randn()
        return action

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    # Display object.
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_width, self.window_size_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        # Check for all possible types of player input.
        for event in pygame.event.get():
            # Command for closing the window.
            if (event.type == pygame.QUIT):
                pygame.quit()
                self.finish = True
                return None
            
            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_DOWN):
                    if self.xRef > 0:
                        self.xRef -= 1.0
                    else:
                        self.xRef = 0.
                    
                elif (event.key == pygame.K_UP):
                    if self.xRef < 12.0:
                        self.xRef += 1.0
                    else:
                        self.xRef = 12.
                    
                elif (event.key == pygame.K_SPACE):
                    self.step(200000*np.random.randn())
        
        # Apply surface over display.
        canvas = pygame.Surface((self.window_size_width, self.window_size_height))
        canvas.fill((255, 255, 255))

        pygame.draw.line(canvas, 'Black', (0, 2*self.baseLine), (self.window_size_width, 2*self.baseLine))
        pygame.draw.line(canvas, 'Black', (0, self.window_size_height - self.baseLine), (self.window_size_width, self.window_size_height - self.baseLine))
        
        # Get position for elevator.
        xElevator = self.window_size_height - int(self._agent_location[0] * 100/3) - 2*self.elevatorHeight
        
        # Get position for counter-weight.
        xCounterWeight = self.window_size_width - int((22 - self._agent_location[0])*100/1.4) + self.baseLine + 2*self.counterWeightHeight
        
        # Display objects.
        pygame.draw.line(canvas, 'Green', (0, int(self.window_size_height - (self.baseLine + self.xRef * 100/3))), (self.window_size_width, int(self.window_size_height - (self.baseLine + self.xRef * 100/3))), width = 2)
        pygame.draw.circle(canvas, 'Black', (self.window_size_width//2, self.baseLine), self.wheelRadius, width = 2)
        pygame.draw.rect(canvas, 'black', (self.window_size_width//2-self.wheelRadius-self.elevatorWidth//2, xElevator, self.elevatorWidth, self.elevatorHeight), width = 2)
        pygame.draw.rect(canvas, 'black', (self.window_size_width//2 + self.wheelRadius-self.counterWeightWidth//2, xCounterWeight, self.counterWeightWidth, self.counterWeightHeight), width = 2)
        pygame.draw.line(canvas, 'black', (self.window_size_width//2-self.wheelRadius, self.baseLine), (self.window_size_width//2 -self.wheelRadius, xElevator), width = 2)
        pygame.draw.line(canvas, 'black', (self.window_size_width//2 + self.wheelRadius, self.baseLine), (self.window_size_width//2 + self.wheelRadius, xCounterWeight), width = 2)
        
        # Limit framerate.
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    # Perform a step.
    def step(self, action):
        self._target_location = self.xRef

        if self.randomActuator:
            action = self.noise_actuator(action)
        x1 = self._agent_location[0]
        x2 = self._agent_location[1]
        x2dot = (action - (self.M - self.m)/self.g - self.b * x2) / (self.M + self.m) 
        self._agent_location[0] = min(max(x1 + self.tau * x2, 0.), 15.)
        self._agent_location[1] = x2 + self.tau * x2dot
        
        # We use `np.clip` to make sure we don't leave the grid
        terminated = np.array_equal(self._agent_location[0], self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Parâmetros do controlador PID.
# M = 500.0
# m = 100.0
# g = 9.8
# b = 0.2

# # Pole position.
# p1 = -5.
# p2 = -2.

# A = np.array([[0, 1], [0, -b/(M+m)]])
# B = np.array([[0], [1/(M+m)]])
# K = place_poles(A, B, [p1, p2])
# K = K.gain_matrix[0]

# # Cria uma especificação de ambiente usando o método gym.spec()
# spec = gym.spec('ElevatorControll-v0')
# # Cria o ambiente personalizado
# env = spec.make(xRef = 5).unwrapped

# # Reseta o ambiente de simulação.
# sensores = env.reset()

# setpoint = []
# posicao  = []

# while True:
#     # Store values.
#     setpoint.append(env.xRef)
#     posicao.append(sensores[0])
    
#     # Renderiza o simulador.
#     env.render()
#     if env.finish:
#         break
    
#     # Aplica a ação de controle.
#     acao = g/(M+m) - K[0]*(sensores[0] - env.xRef) - K[1]*sensores[1]
#     sensores = env.step(acao)
    
# env.close()

# setpoint = np.array(setpoint)
# posicao  = np.array(posicao)
# plt.plot(posicao)
# plt.plot(setpoint)
# plt.show()