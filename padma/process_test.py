import random
from multiprocessing import Process
from time import sleep


def af(q):
    while True:
        print(q)
        sleep(1)


def bf(q):
    while True:
        print(q)
        sleep(2)


def main():
    p = Process(target=af, args=("A",))
    c = Process(target=bf, args=("B",))
    p.start()
    c.start()

    sleep(6)
    print("FIM")
    p.terminate()
    c.terminate()


main()
