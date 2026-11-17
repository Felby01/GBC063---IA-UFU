# agente.py

import random
import copy
from collections import deque

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ambiente import Corrida_Env, RedeDQN

# hiperparâmetros
EPISODIOS = 5000          # quantas sessões de n corridas treinar
GAMMA = 0.95              # peso do futuro (0 = só importa o agora)
LR = 1e-3                 # velocidade de aprendizado
BATCH = 64                # quantas experiências por atualização
MEMORIA_MAX = 10000       # tamanho máximo da memória
MIN_MEMORIA = 500         # só começa a treinar depois de juntar isso
ATUALIZA_ALVO = 200       # a cada quantos passos copia a rede
EPS_INICIO = 1.0          # chance de agir aleatoriamente no começo
EPS_FIM = 0.05
EPS_DECAI = 0.995         # por sessão

# recompensas (ajustáveis)
PESO_PASSO = 1.0              # escala da variação do saldo
PENALIDADE_FALENCIA = -1.0    # quando o saldo chega a zero
BONUS_FINAL = 0.0             # bônus extra no fim (0 = desligado)


def recompensa_do_passo(variacao, saldo_inicial):
    return PESO_PASSO * variacao / saldo_inicial


def decodificar(acao, env):
    n_fracoes = len(env.range_aposta)
    n_podios = len(env.podios)
    cavalo = acao // (n_fracoes * n_podios)
    resto = acao % (n_fracoes * n_podios)
    fracao = resto // n_podios
    segundo_terceiro = env.podios[resto % n_podios]
    # o 1º do palpite é sempre o cavalo apostado
    podio = (cavalo,) + tuple(c for c in segundo_terceiro if c != cavalo)[:2]
    return cavalo, fracao, podio

def escolher_acao(rede, estado, epsilon, n_acoes):
    # exploração: chuta. exploração/uso: usa o que a rede aprendeu
    if random.random() < epsilon:
        return random.randrange(n_acoes)
    with torch.no_grad():
        return int(rede(estado).argmax())


def treinar_passo(rede, alvo, otimizador, memoria):
    lote = random.sample(memoria, BATCH)

    estados = torch.stack([e[0] for e in lote])
    acoes = torch.tensor([e[1] for e in lote])
    recompensas = torch.tensor([e[2] for e in lote], dtype=torch.float32)
    proximos = torch.stack([e[3] for e in lote])
    fins = torch.tensor([e[4] for e in lote], dtype=torch.float32)

    # o que a rede acha que vale cada ação que foi tomada
    q_atual = rede(estados).gather(1, acoes.unsqueeze(1)).squeeze(1)

    # o que deveria valer: recompensa + futuro estimado pela rede alvo
    with torch.no_grad():
        q_proximo = alvo(proximos).max(1).values
        q_alvo = recompensas + GAMMA * q_proximo * (1 - fins)

    perda = nn.functional.smooth_l1_loss(q_atual, q_alvo)
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()
    return perda.item()


def main():
    env = Corrida_Env()
    env.reset()
    n_entradas = len(env.estado())
    n_acoes = env.n_cavalos * len(env.range_aposta) * len(env.podios)

    rede = RedeDQN(n_entradas, n_acoes)
    alvo = copy.deepcopy(rede)
    otimizador = torch.optim.Adam(rede.parameters(), lr=LR)
    memoria = deque(maxlen=MEMORIA_MAX)

    epsilon = EPS_INICIO
    passos = 0
    saldos_treino = []

    # ---------- treino ----------
    for ep in range(EPISODIOS):
        env.reset()
        done = False

        while not done:
            estado = env.estado()
            acao = escolher_acao(rede, estado, epsilon, n_acoes)
            cavalo, fracao, ordem = decodificar(acao, env)

            variacao, done = env.step(cavalo, fracao, ordem)
            r = recompensa_do_passo(variacao, env.saldo_inicial)

            if done:
                if env.saldo <= 0:
                    r += PENALIDADE_FALENCIA
                elif env.saldo > env.saldo_inicial:
                    r += BONUS_FINAL * (env.saldo - env.saldo_inicial) / env.saldo_inicial

            memoria.append((estado, acao, r, env.estado(), float(done)))
            passos += 1

            if len(memoria) >= MIN_MEMORIA:
                treinar_passo(rede, alvo, otimizador, memoria)

            if passos % ATUALIZA_ALVO == 0:
                alvo.load_state_dict(rede.state_dict())

        saldos_treino.append(env.saldo)
        epsilon = max(EPS_FIM, epsilon * EPS_DECAI)

        if (ep + 1) % 200 == 0:
            media = sum(saldos_treino[-200:]) / 200
            print(f"sessão {ep + 1} | média últimas 200: {media:.2f} | epsilon: {epsilon:.2f}")

    # ---------- avaliação ----------
    def agir_aleatorio(env):
        return decodificar(random.randrange(n_acoes), env)

    def agir_dqn(env):
        with torch.no_grad():
            acao = int(rede(env.estado()).argmax())
        return decodificar(acao, env)

    def avaliar(agir, n=2000):
        finais = []
        for _ in range(n):
            env.reset()
            done = False
            while not done:
                cavalo, fracao, ordem = agir(env)
                _, done = env.step(cavalo, fracao, ordem)
            finais.append(env.saldo)
        finais.sort()
        media = sum(finais) / len(finais)
        mediana = finais[len(finais) // 2]
        acima = sum(s > 100 for s in finais) / len(finais)
        return finais, media, mediana, acima

    base, media_b, mediana_b, acima_b = avaliar(agir_aleatorio)
    aprendido, media_d, mediana_d, acima_d = avaliar(agir_dqn)

    print(f"\naleatório | média: {media_b:.2f} | mediana: {mediana_b:.2f} | sessões acima de 100: {acima_b:.1%}")
    print(f"DQN       | média: {media_d:.2f} | mediana: {mediana_d:.2f} | sessões acima de 100: {acima_d:.1%}")

    # ---------- gráficos ----------
    janela = 50
    suavizado = [sum(saldos_treino[max(0, i - janela):i + 1]) / len(saldos_treino[max(0, i - janela):i + 1])
                 for i in range(len(saldos_treino))]

    plt.figure(figsize=(10, 4))
    plt.plot(saldos_treino, alpha=0.3, label="saldo por sessão")
    plt.plot(suavizado, label=f"média móvel ({janela})")
    plt.axhline(100, color="red", linestyle="--", label="saldo inicial")
    plt.yscale("log")
    plt.xlabel("sessão de treino")
    plt.ylabel("saldo final (escala log)")
    plt.title("Aprendizado do DQN")
    plt.legend()
    plt.savefig("aprendizado.png", dpi=150)

    plt.figure(figsize=(10, 4))
    limite = 300
    plt.hist([min(s, limite) for s in base], bins=40, alpha=0.6, label="aleatório")
    plt.hist([min(s, limite) for s in aprendido], bins=40, alpha=0.6, label="DQN")
    plt.axvline(100, color="red", linestyle="--", label="saldo inicial")
    plt.xlabel(f"saldo final da sessão (valores acima de {limite} aparecem em {limite})")
    plt.ylabel("nº de sessões")
    plt.title("Aleatório x DQN")
    plt.legend()
    plt.savefig("comparacao.png", dpi=150)

    torch.save(rede.state_dict(), "rede_dqn.pt")
    return rede

def mostrar_sessao(rede):
    env = Corrida_Env()
    env.reset()
    done = False
    while not done:
        estado = env.estado()
        with torch.no_grad():
            acao = int(rede(estado).argmax())
        cavalo, fracao, ordem = decodificar(acao, env)
        saldo_antes = env.saldo
        _, done = env.step(cavalo, fracao, ordem)
        
        print(f"\ncorrida {env.corrida_atual} | escolheu: {env.nome_do_cavalos[cavalo]} "
              f"| fração: {env.range_aposta[fracao]} | saldo: {saldo_antes:.2f} -> {env.saldo:.2f}")
        ordem_nomes = [env.nome_do_cavalos[i] for i in ordem]
        print(f"  palpite de pódio: {ordem_nomes}")
        print(f"  vencedor: {env.ultimo_ranking[0]} | pódio real: {env.ultimo_ranking[:3]}\n")

if __name__ == "__main__":
    rede_treinada = main()
    mostrar_sessao(rede_treinada)
