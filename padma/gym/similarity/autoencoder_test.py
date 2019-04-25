import time
from threading import Thread

from padma.models.encoder.encoder import EncoderModel
from padma.gym.utils import monta_lista_ids_e_imagens

print('Carregando imagens')
_ids, imagens = monta_lista_ids_e_imagens('2017-07-01', '2017-07-02', 10)
print('%d imagens carregadas.' % len(imagens))
model = EncoderModel()

code = None
cont = 0


def model_predict(image):
    global code
    global cont
    code = model.predict(image)
    cont += 1


# code = model_predict(imagens[0][0])

def faz_thread():
    t = Thread(target=model_predict, args=([imagens[0][0]]))
    t.daemon = True
    t.start()


for r in range(10):
    faz_thread()

tries = 10
while code is None and tries > 1:
    time.sleep(1)
    tries -= 1

print(code)
print(cont)