import csv
import time
import numpy as np
from Setup import InvertedPendulum
import files
from multiprocessing import Process
import matplotlib.pyplot as plt
from matplotlib import animation



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

# Cria o ambiente de simulação.
env = InvertedPendulum(0.50)

# Parâmetros dos controladores PID.
Kp1 = 0
Ki1 = 0
Kd1 = 0
Kp2 = 0.5
Kd2 = 40
Ki2 = 0.0001

# Reseta o ambiente de simulação.
sensores = env.reset()

# Inicialização dos erros e somatórios.
sumErro1 = 0
erroAnt1 = sensores[2]
sumErro2 = 0
erroAnt2 = sensores[0] - env.xRef

########## gráfico ##########

# Cria uma lista vazia para armazenar os dados
dados = []
dadoserro1 = []
dadoserro2 = []
dadosacao = []

Time = 0
dateTime = []

# Cria uma figura e um eixo
fig, ax = plt.subplots()

while True:
    inicio = time.perf_counter()

    if env.user_text_p != '':
        Kp1 = float(env.user_text_p)
    if env.user_text_i != '':
        Ki1 = float(env.user_text_i)
    if env.user_text_d != '':
        Kd1 = float(env.user_text_d)    
    

    # Renderiza o simulador.
    env.render()
    if env.finish:
        break
    
    # Calcula a ação de controle.
    erro1 = sensores[2]
    sumErro1 += erro1
    erro2 = sensores[0] - env.xRef
    sumErro2 += erro2
    acao = (Kp1*erro1 + Kd1*(erro1 - erroAnt1) + Ki1*sumErro1) + (Kp2*erro2 + Kd2*(erro2 - erroAnt2) + Ki2*sumErro2)
    erroAnt1 = erro1
    erroAnt2 = erro2
    
    # Aplica a ação de controle.
    sensores = env.step(acao)
    #Guardar histórico dos dados
    files.WriteFile(a=erro1,b=erro2,c=acao)
            
    #ler os dados armazenados para plotar o gráfico 
    with open('Controle/arquivo.csv', 'r') as arquivo:
        leitor = csv.reader(arquivo)
        dadoserro1 = []
        dadoserro2 = []
        dadosacao = []
        if leitor is not None:
            for linha in leitor:
                tupla = float(linha[0])
                dadoserro1.append(tupla)
                tupla = linha[1]
                dadoserro2.append(tupla)
                tupla = linha[2]
                dadosacao.append(tupla)
    tempo_transcorrido = time.perf_counter() - inicio
    Time +=tempo_transcorrido
    dateTime.append(Time)
    ax.clear
    ax.plot(dateTime, dadoserro1)

env.close()
plt.show()
