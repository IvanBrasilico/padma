import os

path = '/home/ivan/pybr/conteineres_rotulados/'
for filename in os.listdir(path):
    if filename.find('xml') != -1:
        with open(os.path.join(path, filename), 'r') as xml:
            lista = xml.readlines()
        modificado = False
        for ind, linha in enumerate(lista):
            posi = linha.find('/conteiner_to_check')
            if posi != -1:
                print(linha)
                posi = posi + len('/conteiner_to_check')
                linha = '<path>' + path + linha[posi+1:] + '\n'
                print(linha)
                lista[ind] = linha
                modificado = True
                break
            posi = linha.find('<folder>/home/ivan/À')
            if posi != -1:
                print(linha)
                linha = '<folder>'+path+'</folder>\n'
                print(linha)
                lista[ind] = linha
                modificado = True
                break
            posi = linha.find('<folder>conteiner_to_check')
            if posi != -1:
                print(linha)
                linha = '<folder></folder>\n'
                print(linha)
                lista[ind] = linha
                modificado = True
                break
            posi = linha.find('<folder></folder>')
            if posi != -1:
                print(linha)
                linha = '<folder>'+path+'</folder>\n'
                print(linha)
                lista[ind] = linha
                modificado = True
                break
        if modificado:
            with open(os.path.join(path, filename), 'w') as xml:
                lista = xml.writelines(lista)
