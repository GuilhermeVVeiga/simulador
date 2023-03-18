import csv

import numpy as np


def ReadSensor():
    lista_float = list()
    with open('Controle/values.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 1
        for row in reader:
            if i == 1:
                lista = row
                lista_float = [float(x) for x in lista]
                i += 1
    return lista_float


# limpar arquivo do histório de dados
with open('Controle/arquivo.csv', 'w', newline='') as csv_file:
    csv_file.write('')

# escrever o histórico dos dados


def WriteFile(a, b, c):
    with open('Controle/arquivo.csv', 'a', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerow([a, b, c])

# Ler parâmetros de Kp, Ki, Kd


def ReadPID():
    file = open('Controle/PID.txt', 'r')
    content = file.readlines()
    lista_float = [float(x) for x in content]
    file.close()
    return lista_float

# escrever o histórico da acao


def WriteAction(value):
    with open('Controle/acao.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value])

# Escrever valores


def ReadAction():
    force=0
    with open('Controle/acao.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        i = 1
        for row in reader:
            if i == 1:
                force = float(row[0])
                i += 1
    return force

# Escrever valores


def fa(value):
    with open('Controle/values.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(value)
# escrever valores de PID no arquivo


def writeFile(P, I, D):
    file = open("Controle/PID.txt", "w")
    PID = list()
    PID.append(P+"\n")
    PID.append(I+"\n")
    PID.append(D+"\n")
    file.writelines(PID)
    PID.clear()
    file.close()
    return None

def WriteLine(num):
    file = open("Controle/Line.txt", "w")
    file.write(str(num))
    file.close()
    return None

def ReadLine():
    file = open('Controle/Line.txt', 'r')
    content = file.read()
    return float(content)