O script predictions_update e o periodic_updates realizam
consultas ao Servidor PADMA, imagem a imagem, e atualizam o 
Banco de Dados após obter resultado. Para uso online, é
suficiente.

Agora imagine-se que após o BD já possuir mais de 
um milhão de imagens, implanta-se novo modelo. O tempo de
update para este caso seria de dias... Assim, este pacote]
testará e implantará alternativas de rodar predições
 e updates de grandes lotes de imagens. 