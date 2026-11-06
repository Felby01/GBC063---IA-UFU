# Corrida de cavalos com I.A

class Corrida_Env:
    def __init__(self):

        # cavalos
        self.n_cavalos = 5 # quantidade de cavalos
        self.atributos = ["velocidade_maxima", "aceleracao", "vigor"] # identificacao dos atributos

        # corrida
        self.n_corridas = 10 # numeros de corridas na secao
        self.rnd_eve = {"velocidade_aumentada" : 1.25, # eventos aleatorios
                        "lentidao" : 0.75, 
                        "odd_aumentada" : 1.5, 
                        "odd_reduzida" : 0.5, 
                        "vigor_divino" : 1, 
                        "cansaco" : 0.75} 

        # financeiro
        self.saldo_inicial = 100 
        self.limite = 0 # saldo limite pra perder
        self.range_aposta = [0, 0.1, 0.25, 0.50, 0.75, 1] # quantidade disponivel pra aposta de acordo com o saldo atual

        # recompesas
        self.falir = -100 # a qualquer momento
        self.acertar_vencedor = 5 # por corrida
        self.acertar_ordem_e_vencedor = self.acertar_vencedor + 10 # por corrida
        self.terminar_com_saldo_positivo = 100 # no fim da secao
        self.terminar_com_saldo_negativo = -100 # no fim da secao
    
    # inicio da secao

    def reset(self):
        pass

    # cada run
    def step(self, acao):
        pass
