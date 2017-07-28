import getpass
import sys
import telnetlib
import string
import time
import random

HOST = "freechess.org"
user = 'DeepNeuralChess'
password = input("Logging in as " + user + ". Please enter your password: ")
user_bytes = user.encode('utf-8')
password_bytes = password.encode('utf-8')

tn = telnetlib.Telnet(HOST)

tn.read_until(b"login: ")
tn.write(user_bytes + b"\n")
tn.write(password_bytes + b"\n")
tn.read_until(b":")
tn.write(b"\n")
tn.write(b"set tell 1\n")
tn.write(b"set pin 1\n")
tn.write(b"tell 53 Hello, I am a robot!\n")

word_file = "/path-to/random-words.txt"
WORDS = open(word_file).read().splitlines()

while 1:
    cod = input('PRESS P TO PLAY A GAME, OR PRESS Q TO QUIT')
    if cod == 'P':
        tn.write(b"")
    elif cod == 'Q':
        break