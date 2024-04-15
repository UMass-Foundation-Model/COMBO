import argparse
import os
import json
from pathlib import Path
import shutil

import gym
import time
import pickle
import logging
import sys
import numpy as np
# Get the absolute path of the project root directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

gym.envs.registration.register(
	id='tdw_maco-v0',
	entry_point='envs.tdw_gym:TDW'
)

MAX_DATAPOINT = 10000

from agents import CookPlanAgent, GamePlanAgent, COMBOAgent
from utils.utils import convert_np_for_print

class Challenge:
	def __init__(self, logger, task, port, data_path, output_dir, number_of_agents=2, max_steps=30,
				 launch_build=True, screen_size=512, data_prefix='dataset/', save_img=True, save_per_step=8, 
				 skip_success=False, not_skip=False, is_test=False):
		self.env = gym.make("tdw_maco", task=task, port=port, number_of_agents=number_of_agents, save_dir=output_dir,
							max_steps=max_steps, launch_build=launch_build, screen_size=screen_size,
							data_prefix=data_prefix, save_img=save_img, save_per_step=save_per_step, is_test=is_test)
		self.task = task
		self.logger = logger
		self.logger.debug(port)
		self.logger.info("Environment Created")
		self.output_dir = output_dir
		self.max_steps = max_steps
		self.save_img = save_img
		self.skip_success = skip_success
		self.not_skip = not_skip
		self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
		self.logger.info("done")

	def submit(self, agents, logger, eval_episodes, start_id, num_runs):
		if eval_episodes[0] == -1:
			if start_id is not None and num_runs is not None:
				assert start_id + num_runs <= MAX_DATAPOINT, f"datapoint id exceeds {MAX_DATAPOINT}"
				eval_episodes = range(start_id, start_id + num_runs)
			else:
				eval_episodes = range(len(self.data))

		num_eval_episodes = len(eval_episodes)

		start = time.time()
		results = {}
		for i, episode in enumerate(eval_episodes):
			start_time = time.time()
			if not self.skip_success and not self.not_skip:
				if os.path.exists(os.path.join(self.output_dir, str(episode), 'result_episode.json')):
					# The episode has been evaluated before
					with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'r') as f:
						result = json.load(f)
					results[episode] = result
					continue
			elif self.skip_success:
				# only skips successful trials
				path = os.path.join(self.output_dir, str(episode))
				if os.path.exists(path):
					result_path = os.path.join(path, 'result_episode.json')
					if os.path.exists(result_path):
						with open(result_path, 'r') as f:
							result = json.load(f)
						if result['success']:
							results[episode] = result
							continue

				if os.path.exists(path):
					shutil.rmtree(path)

			elif self.not_skip:
				# don't skip
				path = os.path.join(self.output_dir, str(episode))
				if os.path.exists(path):
					shutil.rmtree(path)

			if not os.path.exists(os.path.join(self.output_dir, str(episode))):
				os.makedirs(os.path.join(self.output_dir, str(episode)))
			self.logger.info('Episode {} ({}/{})'.format(episode, i + 1, num_eval_episodes))
			self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
			state, info = self.env.reset(seed=self.data[episode]['seed'], options={
				"output_dir": os.path.join(self.output_dir, str(episode)),
				"save_img": self.save_img,
			})
			self.rng = np.random.RandomState(self.data[episode]['seed'])
			recipe = info['recipe'] if self.task == 'cook' else None
			if self.task == 'cook':
				json.dump(recipe, open(os.path.join(self.output_dir, str(episode), 'recipe.json'), 'w'))

			if self.task == 'game':
				direction = {"direction": "clockwise" if self.env.controller.clockwise else "counter_clockwise"}
				json.dump(direction, open(os.path.join(self.output_dir, str(episode), 'direction.json'), 'w'))

			for agent_id, agent in enumerate(agents):
				if agent.agent_type == 'genco_agent':
					obs = self.filter_obs(state[str(agent_id)])
				else:
					obs = state[str(agent_id)]
				agent.reset(obs, info, output_dir=os.path.join(self.output_dir, str(episode)))
			self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
			done = False
			step_num = 0
			local_reward = 0.0
			metadata = [] # {"step": 0, "actions": "", "frame_start": 0, "frame_end": 13, "prompt": ""}
			camera_matrix_metadata = dict() # dump to pickle
			self.next_agent_id = None
			while not done:
				actions_to_print = {}
				# if self.save_img: self.env.save_images(os.path.join(self.output_dir, str(episode), 'Images'))

				plan_success, actions = self.plan_agent_actions(agents, state)

				if not plan_success:
					done = True
					break

				step_num += 1
				frame_start = self.env.num_frames
				last_obs = convert_np_for_print(state["0"]["objects"])
				state, reward, done, info = self.env.step(actions)
				
				metadata.append({"step": step_num, "obs": last_obs, "actions": actions_to_print, "frame_start": frame_start, "frame_end": self.env.num_frames, "prompt": "", "prompt_value": info["prompt_value"]})
				
				# for agent_id, agent in enumerate(agents):
				# 	print("end_pos: ", agent_id, self.env.controller.agents[agent_id].dynamic.transform.position)
				for agent in agents:
					agent_id = agent.agent_id
					if state[str(agent_id)]["rejected"]:
						final_action = {"type": "wait", "prompt": "wait"}
						if "prompt_proposer" in actions[str(agent_id)]:
							prompt_proposer = actions[str(agent_id)]["prompt_proposer"].split("I choose to")[0]
							final_action["prompt_proposer"] = prompt_proposer + "I choose to wait."

					else:
						final_action = actions[str(agent_id)]

					actions_to_print[str(agent_id)] = convert_np_for_print(final_action)

				metadata[-1]["prompt"] = state["0"]["last_joint_actions"]
				# print(f"metadata: {metadata[-1]['prompt']}\n{metadata[-1]['prompt_value']}")
				# metadata[-1]["frame_end"] += info["num_frames_for_step"]

				if self.save_img:
					for k, v in info["camera_matrices"].items():
						camera_matrix_metadata[k] = v

				local_reward += reward
				self.logger.info(
					f"Executing step {step_num} for episode: {episode}, actions: {actions}, frame: {self.env.num_frames}")
				if done or step_num > self.max_steps:
					break

			if 'success' in info:
				result = {
					"success": info['success'],
					"steps": step_num,
				}
			else:
				result = {
					"success": False,
					"steps": step_num,
				}

			with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'w') as f:
				json.dump(result, f)
			# print(f"metadata: {metadata}")
			with open(os.path.join(self.output_dir, str(episode), 'metadata.json'), 'w') as f:
				json.dump(metadata, f, indent=4)

			# print("camera_matrix_metadata: ", camera_matrix_metadata)
			if self.save_img:
				with open(os.path.join(self.output_dir, str(episode), 'camera_matrix_metadata.pickle'), 'wb') as f:
					pickle.dump(camera_matrix_metadata, f)

			results[episode] = result
		avg_succ = np.mean([results[episode]['success'] for episode in results])
		avg_succ_steps = np.mean([results[episode]['steps'] for episode in results if results[episode]['success']])
		results = {
			"avg_succ": avg_succ,
			"avg_succ_steps": avg_succ_steps,
			"episode_results": results,
		}
		if num_eval_episodes > 1:
			with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
				json.dump(results, f, indent=4)
		self.logger.info(f'eval done, avg success rate {avg_succ}, avg success steps {avg_succ_steps}')
		self.logger.info('time: {}'.format(time.time() - start))
		return avg_succ, avg_succ_steps
	
	def plan_agent_actions(self, agents, state):
		actions = {}
		for agent in agents:
			agent_id = agent.agent_id
			if agent.agent_type == 'combo_agent':
				obs = self.filter_obs(state[str(agent_id)])
			else:
				obs = state[str(agent_id)]
			action = agent.act(obs)
			# print(agent_id, action)
			actions[str(agent_id)] = action
		return True, actions

	def close(self):
		self.env.close()

	def filter_obs(self, obs:dict):
		# filter out oracle observations
		filtered_obs = obs.copy()
		filtered_obs.pop('objects')
		return filtered_obs


def init_logs(output_dir, port, name='simple_example'):
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler(os.path.join(output_dir, f"output_{port}.log"))
	fh.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)
	return logger


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", type=str, choices=("cook", "game", "game_3", "game_2"), default="cook")
	parser.add_argument("--output_dir", type=str, default="results")
	parser.add_argument("--experiment_name", type=str, default="train")
	parser.add_argument("--run_id", type=str, default='run_0')
	parser.add_argument("--data_path", type=str, default="train.json")
	parser.add_argument("--data_prefix", type=str, default="dataset/")
	parser.add_argument("--port", default=1071, type=int)
	parser.add_argument("--start_id", type=int, default=None)
	parser.add_argument("--num_runs", type=int, default=None)
	parser.add_argument("--agents_type", nargs='+', type=str, default=("replicant",))
	parser.add_argument("--agents_algo", nargs='+', type=str, default=("cook_plan_agent", "cook_plan_agent"))
	parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int, help="which episodes to evaluate on")
	parser.add_argument("--max_steps", default=30, type=int, help="max steps per episode")
	parser.add_argument("--no_launch_build", action='store_true')
	parser.add_argument("--debug", action='store_true')
	parser.add_argument("--screen_size", default=512, type=int)
	parser.add_argument("--no_save_img", action='store_true', help="do not save images", default=False)
	parser.add_argument("--save_per_step", default=8, type=int, help="save images every n frames")
	parser.add_argument("--skip_success", action='store_true', help="only skip successful trials", default=False)
	parser.add_argument("--not_skip", action='store_true', help="don't skip episodes", default=False)
	parser.add_argument("--metadata_file", type=str, default="results/game/0/metadata.json" )

	# combo parameters
	parser.add_argument("--only_propose", action="store_true")
	parser.add_argument("--no_belief", action="store_true")
	parser.add_argument("--num_propose", type=int, default=1)
	parser.add_argument("--plan_horizon", type=int, default=1)
	parser.add_argument("--plan_beam", type=int, default=1)

	# LLM parameters
	parser.add_argument("--proposer_lm_id", "-pllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--belief_lm_id", "-bllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--ranker_lm_id", "-rllm", type=str, default="gpt-4-vision-preview")
	parser.add_argument("--lm_source", type=str, choices=["openai", "llava", "azure", "huggingface", "llava_server"], default="openai")
	parser.add_argument("--temperature", "-t", type=float, default=0.7)
	parser.add_argument("--top_p", default=1.0, type=float)
	parser.add_argument("--max_tokens", default=512, type=int)
	parser.add_argument("--n", default=1, type=int)
	parser.add_argument("--logprobs", default=1, type=int)
	parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
	parser.add_argument("--cot", action="store_true")

	parser.add_argument("--guidance_weight", "-gw", type=int, default=5)

	args = parser.parse_args()
	if args.lm_source == 'llava' and 'llava' not in args.proposer_lm_id:
		args.proposer_lm_id = "liuhaotian/llava-v1.5-7b"
		args.belief_lm_id = "liuhaotian/llava-v1.5-7b"
		args.ranker_lm_id = "liuhaotian/llava-v1.5-7b"
	args.number_of_agents = len(args.agents_algo)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.experiment_name)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.task)
	os.makedirs(args.output_dir, exist_ok=True)
	args.output_dir = os.path.join(args.output_dir, args.run_id)
	os.makedirs(args.output_dir, exist_ok=True)
	logger = init_logs(args.output_dir, args.port)

	challenge = Challenge(logger, args.task, args.port, args.data_path, args.output_dir, args.number_of_agents,
						  args.max_steps, not args.no_launch_build, screen_size=args.screen_size,
						  data_prefix=args.data_prefix, save_img=not args.no_save_img, save_per_step=args.save_per_step, 
						  skip_success = args.skip_success, not_skip = args.not_skip,
						  is_test=(args.data_path == "test.json"))
	agents = []
	print(args)
	for i, agent in enumerate(args.agents_algo):
		if agent == 'cook_plan_agent':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=False))
		elif agent == 'cook_plan_agent_altruism':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=True))
		elif agent == 'cook_plan_agent_selfish':
			agents.append(CookPlanAgent(i, logger, args.output_dir, is_altruism=False))
		elif agent == 'game_plan_agent_clockwise': # agent with fixed clockwise passing direction
			agents.append(GamePlanAgent(i, logger, args.output_dir, False, fix_clockwise=True))
		elif agent == 'game_plan_agent_counter_clockwise': # agent with fixed counter-clockwise passing direction
			agents.append(GamePlanAgent(i, logger, args.output_dir, False, fix_clockwise=False))
		elif agent == 'genco_agent':
			agents.append(COMBOAgent(
				task=args.task,
				agent_id=i,
				logger=logger,
				output_dir=args.output_dir,
				max_tokens=args.max_tokens,
				debug_mode=args.debug,
				num_propose=args.num_propose,
				temperature=args.temperature,
				proposer_lm_id=args.proposer_lm_id,
				belief_lm_id=args.belief_lm_id,
				ranker_lm_id=args.ranker_lm_id,
				lm_source=args.lm_source,
				only_propose=args.only_propose,
				no_belief=args.no_belief,
				plan_horizon=args.plan_horizon,
				plan_beam=args.plan_beam,
				cot=args.cot,
				guidance_weight=args.guidance_weight,
			))
		else:
			pass
	try:
		challenge.submit(agents, logger, args.eval_episodes, args.start_id, args.num_runs)
	finally:
		challenge.close()


if __name__ == "__main__":
	main()