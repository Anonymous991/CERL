import torch

# Rollout evaluate an agent in a complete game
def rollout_worker(id, task_pipe, result_pipe, data_bucket, model_bucket):
	"""Rollout Worker runs a simulation in the environment to generate experiences and fitness values

        Parameters:
            task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
            result_pipe (pipe): Sender end of the pipe used to report back results
            is_noise (bool): Use noise?
            data_bucket (list of shared object): A list of shared object reference to s,ns,a,r,done (replay buffer) managed by a manager that is used to store experience tuples
            model_bucket (shared list object): A shared list object managed by a manager used to store all the models (actors)
			env_name (str): Environment name?
			noise_std (float): Standard deviation of Gaussian for sampling noise

        Returns:
            None
    """


	###LOOP###
	while True:
		identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
		if identifier == 'TERMINATE': exit(0) #Kill yourself
		print('Received', identifier)

		net = model_bucket[int(identifier)]
		dummy = torch.ones((10, 100))
		for i in range(100):
			action = net.forward(dummy)


		# Send back id, fitness, total length and shaped fitness using the result pipe
		result_pipe.send(identifier)

