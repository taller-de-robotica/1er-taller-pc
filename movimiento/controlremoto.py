import socket

#UDP_IP = "10.210.0.78"
UDP_IP = "192.168.4.1"
UDP_PORT = 8888

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

comandos = """A - avanzar
L - girar a la izquierda
R - girar a la derecha
B - atrás
E - parar
F - diagonal izquierda arriba
H - diagonal derecha arriba
I - diagonal izquierda abajo
K - diagonal derecha abajo
O - horizontal izquierda
T - horizontal derecha

Q - salir
C - comandos disponibles
"""

print(comandos)

while True:
    c = input("Envíe un comando: ")
    if c == 'Q' or c == 'q':
        break
    elif c == 'C' or c == 'c':
        print(comandos)
    mes = bytes(c, 'ascii')
    sock.sendto(mes, (UDP_IP, UDP_PORT))
