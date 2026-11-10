# Corrida de cavalos com I.A

import random
class Corrida_Env:
    
    # regras do ambiente
    def __init__(self):

        # cavalos
        self.n_cavalos = 5 # quantidade de cavalos
        self.atributos = ["velocidade_maxima", "aceleracao", "vigor"] # identificacao dos atributos
        self.nome_do_cavalos = ["Cavalo Azul", "Cavalo Verde", "Cavalo Vermelho", "Cavalo Amarelo", "Cavalo Roxo"]

        # corrida
        self.n_corridas = 10 # numeros de corridas na secao
        self.rnd_eve = {"velocidade_aumentada" : 1.25, # eventos aleatorios
                        "lentidao" : 0.75, 
                        "odd_aumentada" : 1.5, 
                        "odd_reduzida" : 0.5, 
                        "vigor_divino" : 1, 
                        "cansaco" : 0.75} 

        # financeiro
        self.range_aposta = [0, 0.1, 0.25, 0.50, 0.75, 1] # quantidade disponivel pra aposta de acordo com o saldo atual

        # recompesas
        self.falir = 0 # a qualquer momentohe
        self.acertar_vencedor = 5 # por corrida
        self.acertar_ordem_e_vencedor = self.acertar_vencedor + 10 # por corrida
        self.terminar_com_saldo_positivo = 100 # no fim da secao
        self.terminar_com_saldo_negativo = 0 # no fim da secao

        self.pontos_totais = 100
        self.pontos_minimos = 25
    
    # inicio da secao

    def reset(self):

        saldo_inicial = 100
        corrida_atual = 1
        self.cavalos = []
        extra = self.pontos_totais - self.pontos_minimos * len(self.atributos)

        for i in range(self.n_cavalos):
           
            sorteio = [random.random() ** 2 for _ in self.atributos]
            soma = sum(sorteio)

           
            cavalo = {"nome": self.nome_do_cavalos[i], "cansaco" : 0}

           
            for j in range(len(self.atributos)):
                nome_atributo = self.atributos[j]
                cavalo[nome_atributo] = round(self.pontos_minimos + extra * sorteio[j] / soma,2)

            media = self.pontos_totais/(cavalo["velocidade_maxima"] * 0.8 + cavalo["aceleracao"] * 0.2 +  cavalo["vigor"] * 0.5 )      
            cavalo["odd"] = round(media, 2)
            
            self.cavalos.append(cavalo)

        # ajustando a odd

        total = 0
        
        for k in range(self.n_cavalos):
            total += cavalo["odd"]

       



            

    # cada run
    def step(self, acao):
        pass

env = Corrida_Env()
env.reset()
for c in env.cavalos:
    print(c)