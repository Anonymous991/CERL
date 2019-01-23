from core.off_policy_algo import Off_Policy_Algo, SAC




class Learner:
	"""Learner object encapsulating a local learner

		Parameters:
		algo_name (str): Algorithm Identifier
		state_dim (int): State size
		action_dim (int): Action size
		actor_lr (float): Actor learning rate
		critic_lr (float): Critic learning rate
		gamma (float): DIscount rate
		tau (float): Target network sync generate
		init_w (bool): Use kaimling normal to initialize?
		**td3args (**kwargs): arguments for TD3 algo


	"""

	def __init__(self, wwid, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w = True, **td3args):
		self.td3args = td3args; self.id = id
		self.algo = Off_Policy_Algo(wwid, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w)


		#LEARNER STATISTICS
		self.fitnesses = []
		self.ep_lens = []
		self.value = None
		self.visit_count = 0


	def update_parameters(self, replay_buffer, buffer_gpu, batch_size, iterations):
		for _ in range(iterations):
			s, ns, a, r, done = replay_buffer.sample(batch_size)
			if not buffer_gpu:
				s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); done = done.cuda()
			self.algo.update_parameters(s, ns, a, r, done, 1, **self.td3args)


	def update_stats(self, fitness, ep_len, gamma=0.2):
		self.visit_count += 1
		self.fitnesses.append(fitness)
		self.ep_lens.append(ep_len)

		if self.value == None: self.value = fitness
		else: self.value = gamma * fitness + (1-gamma) * self.value
