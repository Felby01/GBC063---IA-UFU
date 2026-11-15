# Corrida de cavalos com I.A
# Bibliotecas
import random
import torch
import torch.nn as nn
import itertools

# Ambiente da corrida
class Corrida_Env:

    # regras do ambiente
    def __init__(self):

        # cavalos
        self.nome_do_cavalos = ["Cavalo Azul", "Cavalo Verde", "Cavalo Vermelho",
                                "Cavalo Amarelo", "Cavalo Roxo", "Cavalo Branco"]
        self.n_cavalos = len(self.nome_do_cavalos)
        self.atributos = ["velocidade_maxima", "aceleracao", "vigor"]

        # corrida
        self.n_corridas = 10
        # valor de cada evento: multiplicador de força ou de odd, ou pontos de cansaço
        # vigor_divino não usa o valor: só segura cansaço e desgaste do cavalo afetado
        self.rnd_eve = {"velocidade_aumentada": 1.25,
                        "lentidao": 0.75,
                        "odd_aumentada": 1.5,
                        "odd_reduzida": 0.5,
                        "vigor_divino": 1,
                        "cansaco": 30}
        self.ordens = list(itertools.permutations(range(self.n_cavalos)))  # 720 ordens possíveis
        self.podios = list(itertools.permutations(range(self.n_cavalos), 3))  # 120 pódios possíveis

        # fadiga
        self.cansaco_por_corrida = 10   # antes do ajuste pelo vigor
        self.recuperacao = 0.5          # fração do cansaço que volta entre as corridas
        self.desgaste_por_corrida = 1   # antes do ajuste pelo vigor

        # financeiro
        self.saldo_inicial = 100
        self.range_aposta = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1]
        self.tabela_odds = [1.8, 2.8, 5.5, 12, 20, 40]   # tem que ter um valor por cavalo
        self.bonus_ordem = 0.5                           # bônus = 50% da aposta se acertar a ordem

        # recompensas (ainda não usadas)
        self.falir = 0
        self.acertar_vencedor = 5
        self.acertar_ordem_e_vencedor = self.acertar_vencedor + 10
        self.terminar_com_saldo_positivo = 100
        self.terminar_com_saldo_negativo = 0

        self.pontos_totais = 100
        self.pontos_minimos = 25

    # inicio da secao
    def reset(self):
        self.saldo = self.saldo_inicial
        self.corrida_atual = 0
        self.ultima_nao_apostou = False
        self.cavalos = []
        extra = self.pontos_totais - self.pontos_minimos * len(self.atributos)

        # cria os cavalos
        for i in range(self.n_cavalos):
            sorteio = [random.random() ** 2 for _ in self.atributos]
            soma = sum(sorteio)

            cavalo = {"nome": self.nome_do_cavalos[i], "cansaco": 0, "desgaste": 0}

            for j in range(len(self.atributos)):
                nome_atributo = self.atributos[j]
                cavalo[nome_atributo] = round(self.pontos_minimos + extra * sorteio[j] / soma, 2)

            # média ponderada para definir a força de cada cavalo
            cavalo["forca"] = round(cavalo["velocidade_maxima"] * 0.8
                                    + cavalo["aceleracao"] * 0.2
                                    + cavalo["vigor"] * 0.5, 2)

            self.cavalos.append(cavalo)

        # cada cavalo conta quantos são mais fortes que ele
        for cavalo in self.cavalos:
            posicao = 0
            for outro in self.cavalos:
                if outro["forca"] > cavalo["forca"]:
                    posicao += 1
            cavalo["odd"] = self.tabela_odds[posicao]

        # sorteia o evento da primeira corrida
        self.sortear_corrida()

    # sorteia o evento da próxima corrida e quem será afetado
    # o agente vê o evento, mas não vê os afetados
    def sortear_corrida(self):
        self.evento_atual = random.choice(list(self.rnd_eve))
        nomes = [c["nome"] for c in self.cavalos]
        self.afetados = random.sample(nomes, self.n_cavalos // 2)

    # força do cavalo naquela corrida, já com evento e fadiga
    def forca_na_corrida(self, cavalo):
        forca = cavalo["forca"]
        afetado = cavalo["nome"] in self.afetados

        # velocidade aumentada ou lentidão mudam a força só dos afetados
        if afetado and self.evento_atual in ("velocidade_aumentada", "lentidao"):
            forca *= self.rnd_eve[self.evento_atual]

        # cansaço e desgaste reduzem a força, exceto com fôlego divino
        if not (afetado and self.evento_atual == "vigor_divino"):
            fadiga = cavalo["cansaco"] + cavalo["desgaste"]
            forca *= max(0.1, 1 - fadiga / 100)

        return forca

    # odd que será paga naquela corrida
    def odd_da_corrida(self, cavalo):
        odd = cavalo["odd"]
        if cavalo["nome"] in self.afetados and self.evento_atual in ("odd_aumentada", "odd_reduzida"):
            odd *= self.rnd_eve[self.evento_atual]
        return odd

    # corridas
    def correr(self):
        pontos = {}

        for cavalo in self.cavalos:
            sorte = random.uniform(0.9, 1.1)
            pontos[cavalo["nome"]] = self.forca_na_corrida(cavalo) * sorte

        ranking = sorted(pontos, key=pontos.get, reverse=True)
        return ranking

    # depois da corrida: cansaço sobe com pouco vigor, desgaste acumula
    def atualizar_fadiga(self):
        for cavalo in self.cavalos:
            afetado = cavalo["nome"] in self.afetados
            if afetado and self.evento_atual == "vigor_divino":
                continue   # segura: não ganha cansaço, não recupera, não desgasta

            fator = 1 - cavalo["vigor"] / 100   # mais vigor, menos desgaste
            cavalo["cansaco"] = cavalo["cansaco"] * self.recuperacao + self.cansaco_por_corrida * fator
            cavalo["desgaste"] += self.desgaste_por_corrida * fator

    # uma aposta e uma corrida
    def step(self, cavalo_escolhido, acao_fracao, ordem_prevista=None):

        # obriga a apostar a menor faixa diferente de zero
        if self.ultima_nao_apostou and acao_fracao == 0:
            acao_fracao = 1
        self.ultima_nao_apostou = (acao_fracao == 0)

        saldo_antes = self.saldo

        # decide quanto apostar
        fracao = self.range_aposta[acao_fracao]
        aposta = self.saldo * fracao
        self.ultima_aposta = aposta
        self.saldo -= aposta

        # evento de cansaço: aplica nos afetados antes da corrida
        if self.evento_atual == "cansaco":
            for cavalo in self.cavalos:
                if cavalo["nome"] in self.afetados:
                    cavalo["cansaco"] += self.rnd_eve["cansaco"]

        # roda a corrida e guarda o ranking que decidiu o resultado
        ranking = self.correr()
        self.ultimo_ranking = ranking
        cavalo = self.cavalos[cavalo_escolhido]

        # acertou o vencedor: ganha a odd (já com evento)
        if ranking[0] == cavalo["nome"]:
            self.saldo += aposta * self.odd_da_corrida(cavalo)

            # acertou também o pódio (1º, 2º e 3º): ganha o bônus
            if ordem_prevista is not None:
                ordem_real = [self.nome_do_cavalos.index(nome) for nome in ranking[:3]]
                if tuple(ordem_prevista) == tuple(ordem_real):
                    self.saldo += aposta * self.bonus_ordem
        # fadiga depois da corrida
        self.atualizar_fadiga()

        # avança a corrida e calcula a recompensa
        self.corrida_atual += 1
        recompensa = self.saldo - saldo_antes

        # verifica se a sessão acabou
        done = self.corrida_atual >= self.n_corridas or self.saldo <= 0

        # se continua, já sorteia o evento da próxima corrida
        if not done:
            self.sortear_corrida()

        return recompensa, done

    # o que o agente enxerga (sem a força e sem os afetados)
    def estado(self):
        lista = []
        for c in self.cavalos:
            lista += [c["velocidade_maxima"] / 100, c["aceleracao"] / 100, c["vigor"] / 100,
                      c["cansaco"] / 100, c["desgaste"] / 100, c["odd"] / 40]
        lista += [self.saldo / self.saldo_inicial, self.corrida_atual / self.n_corridas]
        lista += [1.0 if e == self.evento_atual else 0.0 for e in self.rnd_eve]
        return torch.tensor(lista, dtype=torch.float32)


# Rede DQN
class RedeDQN(nn.Module):
    def __init__(self, n_entradas, n_acoes):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Linear(n_entradas, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_acoes),
        )

    def forward(self, x):
        return self.camadas(x)

if __name__ == '__main__':
    # Teste 1: 3 corridas com evento anunciado
    env = Corrida_Env()
    env.reset()

    for rodada in range(1, 4):
        print(f"\n--- corrida {rodada} ---\n")
        print("evento desta corrida:", env.evento_atual)
        print("afetados (só para conferir):", env.afetados)

        cavalo = random.randint(0, env.n_cavalos - 1)
        acao = random.randint(0, len(env.range_aposta) - 1)
        ordem = random.sample(range(env.n_cavalos), 3)

        recompensa, done = env.step(cavalo, acao, ordem)

        print("cavalo escolhido: ", env.nome_do_cavalos[cavalo], "| valor apostado: ", round(env.ultima_aposta, 2))
        print("ordem real:", env.ultimo_ranking)
        print("saldo:", round(env.saldo, 2), "| recompensa:", round(recompensa, 2), "| done:", done)

    # Teste 2: formato do estado para a rede
    x = env.estado()
    rede = RedeDQN(len(x), env.n_cavalos * len(env.range_aposta) * len(env.podios))
    print("entradas:", len(x), "| saídas:", rede(x).shape)