import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

actual_date = datetime.now().date()
algo = "PPO"
##### homogéneo #####

# with open('trained_policy_gridworld\default_experiment-2025-02-18-08-36-41-8c70b62450fa40f18070cf08da2e7aa7\learning_curves maddpg 04-22\default_experiment_rewards.pkl', 'rb') as f:
#     rew_data = pickle.load(f)
# with open('trained_policy_gridworld\default_experiment-2025-02-18-08-36-41-8c70b62450fa40f18070cf08da2e7aa7\learning_curves maddpg 04-22\default_experiment_ploss.pkl', 'rb') as f:
#     ploss_data = pickle.load(f)
# with open(
#         'trained_policy_gridworld\default_experiment-2025-02-18-08-36-41-8c70b62450fa40f18070cf08da2e7aa7\learning_curves maddpg 04-22\default_experiment_qloss.pkl', 'rb') as f:
#     qloss_data = pickle.load(f)
# with open(
#         'trained_policy_gridworld\default_experiment-2025-02-18-08-36-41-8c70b62450fa40f18070cf08da2e7aa7\learning_curves maddpg 04-22\default_experiment_vvio.pkl', 'rb') as f:
#     vvio_data = pickle.load(f)
# plt.title("rew homogéneo")
# plt.plot(rew_data, label='Rewards')
# plt.legend(loc='best')
# plt.show()
# plt.title("loss homogéneo")
# plt.plot(ploss_data, label='P_Loss')
# plt.plot(qloss_data, label='Q_loss')
# #plt.plot(vvio_data, label='VVio')E
# plt.legend(loc='best')
# plt.show()

##### heterogéneo #####

with open('C:\\Users\\Usuario\\Documents\\Programas\\MARLlib\\examples\\exp_results\\ippo_mlp_Checkers\IPPOTrainer_magym_Checkers_6004d_00000_0_2025-03-11_10-11-42\\result.json',
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

##### mappo-->['info']['learner']['default_policy']['learner_stats']['total_loss']
#### ipg-->['info']['learner']['shared_policy']['learner_stats']['policy_loss']
#Sacar tipos de agentes (nombres)
agents = []
# for agent in enumerate(data[0]['policy_reward_mean']):
#     agents.append(agent[1])

for episode in range(len(data)): # TODO: Está hecho para 3 agentes
    loss_episode.append(data[episode]['info']['learner']['default_policy']['learner_stats']['total_loss'])
    # loss_episode_ag0.append(data[episode]['info']['learner'][agents[0]]['total_loss']) #vf_loss_unclipped
    # loss_episode_ag1.append(data[episode]['info']['learner'][agents[1]]['total_loss']) #total_loss
    # loss_episode_ag2.append(data[episode]['info']['learner'][agents[2]]['total_loss'])

    reward_episode.append(data[episode]['episode_reward_mean'])
    # reward_episode_ag0.append(data[episode]['policy_reward_mean'][agents[0]])
    # reward_episode_ag1.append(data[episode]['policy_reward_mean'][agents[1]])
    # reward_episode_ag2.append(data[episode]['policy_reward_mean'][agents[2]])

axl.set_title('loss')
axl.plot(range(0, len(loss_episode)), np.asarray(loss_episode), label='total_loss')
# axl.plot(range(0, len(loss_episode_ag0)), np.asarray(loss_episode_ag0), label=f'loss_{agents[0]}')
# axl.plot(range(0, len(loss_episode_ag1)), np.asarray(loss_episode_ag1), label=f'loss_{agents[1]}')
# axl.plot(range(0, len(loss_episode_ag2)), np.asarray(loss_episode_ag2), label=f'loss_{agents[2]}')
axl.legend(loc='best')
# axl.set_xlim(50, )
# axl.set_ylim(0, 100000)
#figl.savefig(f'Loss_{actual_date}_{algo}.png', dpi=600)
figl.show()

axr.set_title('reward')
axr.plot(range(0, len(reward_episode)), np.asarray(reward_episode), label='total_rew')
# axr.plot(range(0, len(reward_episode_ag0)), np.asarray(reward_episode_ag0), label=f'rew_{agents[0]}')
# axr.plot(range(0, len(reward_episode_ag1)), np.asarray(reward_episode_ag1), label=f'rew_{agents[1]}')
# axr.plot(range(0, len(reward_episode_ag2)), np.asarray(reward_episode_ag2), label=f'rew_{agents[2]}')
axr.legend(loc='best')
# axr.set_xlim(50, )
# axr.set_ylim(-1000, 0)
#figr.savefig(f'curvas/Reward{actual_date}_{algo}.png', dpi=600)
figr.show()
