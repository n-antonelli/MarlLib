import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

actual_date = datetime.now().date()
algo = "PPO"

algoritmo = "mappo_mlp_case33_3min_final"
dirección = {
    "mappo" : ['shared_policy','total_loss'],
    "ippo" : ['shared_policy','total_loss'],
    "vdppo" : ['shared_policy','total_loss'],
    "vda2c" : ['shared_policy','grad_gnorm'],
}
##### heterogéneo #####

with open('C:\\Users\\Usuario\\Documents\\Programas\\MARLlib\\examples\\exp_results\\vdppo_mlp_case33_3min_final\VDPPOTrainer_voltage_case33_3min_final_1ef21_00000_0_2025-03-12_11-56-04\\result.json',
          'r') as file:
    data = []
    for episode in file:
        data.append(json.loads(episode))

figl, axl = plt.subplots()
loss_episode = []
loss_episode_ag0 = []
loss_episode_ag1 = []
loss_episode_ag2 = []
figr, axr = plt.subplots()
reward_episode = []
reward_episode_ag0 = []
reward_episode_ag1 = []
reward_episode_ag2 = []

#Sacar tipos de agentes (nombres)

for episode in range(len(data)): # TODO: Está hecho para 3 agentes
    loss_episode.append(data[episode]['info']['learner']['shared_policy']['learner_stats']['total_loss'])

    reward_episode.append(data[episode]['episode_reward_mean'])

axl.set_title(f'{algoritmo} loss')
axl.plot(range(0, len(loss_episode)), np.asarray(loss_episode), label='total_loss')

axl.legend(loc='best')
# axl.set_xlim(50, )
# axl.set_ylim(0, 100000)
#figl.savefig(f'Loss_{actual_date}_{algo}.png', dpi=600)
figl.show()

axr.set_title(f'{algoritmo} reward')
axr.plot(range(0, len(reward_episode)), np.asarray(reward_episode), label='total_rew')

axr.legend(loc='best')
# axr.set_xlim(50, )
# axr.set_ylim(-1000, 0)
#figr.savefig(f'curvas/Reward{actual_date}_{algo}.png', dpi=600)
figr.show()
