import gym
from gym.envs.registration import register

from environment.standard_env import PressDiplomacyEnv
from inferred_agent.agent import AGENT
from inferred_agent.nego_model import NEGOTIATOR
from inferred_agent.trust_model import TRUSTER
from inferred_agent.decision_model import ORDERER
from environment.l1_negotiation import Negotiation
from environment.order import Order

if __name__ == '__main__':

    env = PressDiplomacyEnv()
    env.reset()
    nego_module = NEGOTIATOR['dipblue']
    order_module = ORDERER['dipblue']
    trust_module = TRUSTER['dipblue']
    agent = AGENT['dipblue']
    weights = [0.5, 0.5, 0, 0.3, 0.2, 0.1, 0.4]
    powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
    players = {p:agent(env.static_infos, me=p, weights=w) for p, w in zip(powers, weights)}

    order_parser = Order()
    nego_parser = Negotiation()

    # Update Modules
    for power_name, player in players.items():
        player.orderer = order_module(env.static_infos, power_name, player.weights)
        player.advisers = player.orderer.advisers #TODO : functionalize the adviser
        player.negotiator = nego_module(player.strategy_yes, player.strategy_balanced,
                                        powers, power_name, player.static_dict['loc_abut'])
        player.truster = trust_module(power_name, order_parser, nego_parser, powers)

    obs, infos = env.reset()
    import time
    start =time.time()
    while env.is_done() is False:
        print(infos['name'])
        if not env.is_nego:
            #start = time.time()
            for power in powers:
                power_orders, _, prev_order_clear = players[power].act(infos)
                #power_orders = [random.choice(valid_actions[loc]) for loc in
                #                env.game.get_orderable_locations(power)]
                env.submit((power, power_orders), prev_order_clear)
            obs, rew, dones, infos = env.step(None)
            #print('Order Iteration :',time.time() - start)
        else:
            #start = time.time()
            for power in powers:
                negos, agreed_orders, _ = players[power].act(infos)
                env.submit(negos, agreed_orders)
            obs, rew, done, infos = env.step(None)
            #print('Nego Iteration :',time.time() - start)
    print('One Game  Iteration :', time.time() - start)







