import gymnasium  as gym
from gymnasium import spaces
import gym_env
import numpy as np
from scipy.signal import place_poles

# Parâmetros do sistema.
g = 9.8
M = 1.0
m = 0.1
l = 0.5
L = 2*l
I = m*L**2 / 12

# SENSORES.
# sensores[0]: posição.
# sensores[1]: velocidade.
# sensores[2]: ângulo.
# sensores[3]: velocidade angular.
# SETPOINT em env.xRef.

# Cria uma especificação de ambiente usando o método gym.spec()
###########         ambeintes        ###########
# spec = gym.spec('gym_env/InvertedPendulum-v22')
spec = gym.spec('gym_env/Elevator-v29')

##########################################

# Cria o ambiente personalizado
env = spec.make().unwrapped

observation,_ = env.reset()
sensores = observation['agent']

# Parâmetros dos controladores PID.
ang_old = 0
pos_old = 0

# Parâmetros do controlador PID.
M = 500.0
m = 100.0
g = 9.8
b = 0.2

# Pole position.
p1 = -5.
p2 = -2.

A = np.array([[0, 1], [0, -b/(M+m)]])
B = np.array([[0], [1/(M+m)]])
K = place_poles(A, B, [p1, p2])
K = K.gain_matrix[0]

def funcao_controle_3(sensores):
    kp_pos=90
    kd_pos=5500
    kp_ang=200
    kd_ang=5000
    acao = kp_pos*(sensores[0]-env.xRef) + kd_pos*(sensores[0] - pos_old) + kp_ang*sensores[2] + kd_ang*(sensores[2] - ang_old)
    
    return acao

while True:
    env.render()
    if env.finish:
        break
    acao = g/(M+m) - K[0]*(sensores[0] - env.xRef) - K[1]*sensores[1]
    # acao = funcao_controle_3(sensores)
    print(acao)
    ang_old = sensores[2]
    pos_old = sensores[0] 
    observation, reward, terminated, condition, info = env.step(acao)
    sensores = observation['agent']
    
env.close()