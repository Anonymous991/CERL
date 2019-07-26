
from core.models import Actor
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from core.buffer import Buffer








class CERL_Agent:
	"""Main CERL class containing all methods for CERL

		Parameters:
		args (int): Parameter class with all the parameters

	"""

	def __init__(self):
		popsize = 10

		#MP TOOLS
		self.manager = Manager()

		#Initialize population
		self.pop = self.manager.list()
		for _ in range(popsize):
			self.pop.append(Actor(100,10, -1))

		#Turn off gradients and put in eval mod
		for actor in self.pop:
			actor = actor.cpu()
			actor.eval()


		# Initialize shared data bucket
		self.replay_buffer = Buffer(1000000, False)
		self.data_bucket = self.replay_buffer.tuples


		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(popsize)]
		self.evo_result_pipes = [Pipe() for _ in range(popsize)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], self.data_bucket, self.pop))
							for id in range(popsize)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(popsize)]



	def train(self):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############

		for id, actor in enumerate(self.pop):
			self.evo_task_pipes[id][0].send(str(id))




		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		for i in range(len(self.pop)):
			entry = self.evo_result_pipes[i][1].recv()
			print ('Returned', entry)



		return







if __name__ == "__main__":



	#INITIALIZE THE MAIN AGENT CLASS
	agent = CERL_Agent() #Initialize the agent

	for gen in range(1, 1000000000): #Infinite generations
		agent.train()
		print('Gen', gen)


