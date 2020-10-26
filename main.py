import environment as env
import random
import time

def calc_loss(rewards):
    loss = 0
    for i in range(len(rewards)):
        loss += abs(1 - rewards[i]) * i * 0.001
    return loss


env.reset()
observation, loss, done, heart_is_use = env.step(0)

"""
env.step принимает action: 1 - биться сердцу, 0 - не биться сердцу.
         возвращает:
            - observation: list, частота биения сердца и объём крови;
            - done: bool, True, если мы сердце бьётся, False - если нет;
            - heart_is_use: bool. Тут поподробнее...
                если предсердия уже начали сокращаться, то мы никак на это не повлияем,
                да и смысла в этом нет. Поэтому мы должны подавать информацию нейросети,
                если heart_is_use имеет значение False.
"""
rewards = []
while True:
    time.sleep(0.01)
    if not heart_is_use:
        action = 1 if random.randint(0, 400) > 390 else 0 # action должен генерироваться нейросетью
        observation, reward, done, heart_is_use = env.step(action)
        rewards.append(reward)
    else:
        _, _, _, heart_is_use = env.step(0)
    
    env.render()
    
    if done:
        print(calc_loss(rewards))
        rewards = []
        env.reset()