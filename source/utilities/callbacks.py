import csv
import sys
import os

import pandas as pd

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

FILE_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIRECTORY: str = os.path.dirname(FILE_DIRECTORY)
PROJECT_ROOT: str = os.path.dirname(SOURCE_DIRECTORY)

if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from source.utilities.helpers import cleanup_checkpoints

class TensorboardCallback(BaseCallback):
	def __init__(self,
               episode_path: str = None,
               first_episode_logged: bool = False,
               verbose: int = None) -> None:
		if episode_path is None:
			raise ValueError("Episode path not specified.")
		if first_episode_logged is None:
			raise ValueError("First episode logged not specified.")
		if verbose is None:
			raise ValueError("Verbose level not specified.")

		super().__init__(verbose = verbose)

		self.episode_path: str = episode_path
		self.first_episode_logged: bool = first_episode_logged
		self.episode_count: int = 0
		self.total_timesteps: int = 0
		self.success_count: int = 0
		self.failure_count: int = 0
	def _init_callback(self) -> None:
		if self.first_episode_logged:
			try:
				data: pd.DataFrame = pd.read_csv(self.episode_path)
				self.episode_count = data['episode'].iloc[-1]
				self.total_timesteps = data['total_timesteps'].iloc[-1]
				self.success_count = data['success'].sum()
				self.failure_count = data['failure'].sum()
			except Exception as exception:
				print(f"Warning: Could not read existing CSV for TensorboardCallback, starting fresh: {exception}")
	def _on_step(self) -> bool:
		informations: list = self.locals.get('infos')

		for information in informations:
			episode: dict = information.get('episode')

			if episode:
				if information.get('success') == 1:
					self.success_count += 1
				if information.get('failure') == 1:
					self.failure_count += 1

				length: int = episode.get('l')
				reward: float = episode.get('r')
				alpha: float = information.get('alpha')
				total_moves: int = information.get('total_moves')
				total_senses: int = information.get('total_senses')
				success: int = information.get('success')
				failure: int = information.get('failure')
				self.episode_count += 1
				self.total_timesteps += length
				total_actions: int = total_moves + total_senses
				move_ratio: float = float(total_moves) / float(total_actions)
				sense_ratio: float = float(total_senses) / float(total_actions)
				success_rate: float = float(self.success_count) / float(self.episode_count) * 100.0
				failure_rate: float = float(self.failure_count) / float(self.episode_count) * 100.0

				self.logger.record('time/episode_count',
                           self.episode_count)
				self.logger.record('time/total_timesteps',
                       		 self.total_timesteps)
				self.logger.record('episode/length',
                       		 length)
				self.logger.record('episode/reward',
                       		 reward)
				self.logger.record('custom/alpha',
											 		 alpha)
				self.logger.record('custom/total_moves',
                       		 total_moves)
				self.logger.record('custom/total_senses',
                       		 total_senses)
				self.logger.record('custom/success',
                       		 success)
				self.logger.record('custom/failure',
                       		 failure)
				self.logger.record('derived/move_ratio',
                       		 move_ratio)
				self.logger.record('derived/sense_ratio',
                       		 sense_ratio)
				self.logger.record('performance/success_rate',
                       		 success_rate)
				self.logger.record('performance/failure_rate',
                       		 failure_rate)

		return True
class EpisodeCallback(BaseCallback):
	def __init__(self,
               episode_path: str = None,
               episodes: int = None,
               first_episode_logged: bool = False,
               verbose: int = None) -> None:
		if episode_path is None and episodes is None:
			raise ValueError("Either episode path or episodes must be specified.")
		if first_episode_logged is None:
			raise ValueError("First episode logged not specified.")
		if verbose is None:
			raise ValueError("Verbose level not specified.")

		super().__init__(verbose = verbose)

		self.episode_path: str = episode_path
		self.episodes: int = episodes
		self.first_episode_logged: bool = first_episode_logged
		self.episode_count: int = 0
		self.total_timesteps: int = 0
		self.success_count: int = 0
		self.failure_count: int = 0
		self.progress_bar: tqdm = None
	def _init_callback(self) -> None:
		if self.first_episode_logged:
			try:
				data: pd.DataFrame = pd.read_csv(self.episode_path)
				self.episode_count = data['episode'].iloc[-1]
				self.total_timesteps = data['total_timesteps'].iloc[-1]
				self.success_count = int(data['success'].sum())
				self.failure_count = int(data['failure'].sum())
			except Exception as exception:
				print(f"Warning: Could not read existing CSV for EpisodeCallback, starting fresh: {exception}")
		else:
			with open(self.episode_path,
                "w",
                encoding = "utf-8",
                newline = "") as file:
				writer: csv.writer = csv.writer(file)

				writer.writerow(["episode",
												 "total_timesteps",
												 "length",
												 "reward",
												 "alpha",
												 "total_moves",
												 "move_ratio",
												 "total_senses",
												 "sense_ratio",
												 "success",
												 "success_rate",
												 "failure",
												 "failure_rate"])

		if self.episodes:
			self.progress_bar = tqdm(total = self.episodes,
                               desc = "Evaluation")
	def log_episode(self, 
									data: dict = None) -> None:
		if data is None:
			raise ValueError("Data not specified.")

		length: int = data.get('length')
		reward: float = data.get('reward')
		alpha: float = data.get('alpha')
		total_moves: int = data.get('total_moves')
		total_senses: int = data.get('total_senses')
		success: int = data.get('success')
		failure: int = data.get('failure')

		if success:
			self.success_count += 1
		elif failure:
			self.failure_count += 1
		else:
			raise ValueError("Episode must be marked as success or failure.")

		self.episode_count += 1
		self.total_timesteps += length
		total_actions: int = total_moves + total_senses
		move_ratio: float = float(total_moves) / float(total_actions)
		sense_ratio: float = float(total_senses) / float(total_actions)
		success_rate: float = float(self.success_count) / float(self.episode_count)
		failure_rate: float = float(self.failure_count) / float(self.episode_count)
		row: list = [self.episode_count,
								 self.total_timesteps,
								 length,
								 reward,
								 alpha,
								 total_moves,
								 move_ratio,
								 total_senses,
								 sense_ratio,
								 success,
								 success_rate,
								 failure,
								 failure_rate]

		if self.episode_path:
			with open(self.episode_path,
                "a",
                encoding = "utf-8",
                newline = "") as file:
				csv.writer(file).writerow(row)
		if self.progress_bar:
			self.progress_bar.update(1)
	def get_data(self,
							 information: dict = None) -> dict:
		if information is None:
			raise ValueError("Information not specified.")

		episode: dict = information.get('episode')
		length: int = episode.get('l')
		reward: float = episode.get('r')
		alpha: float = information.get('alpha')
		total_moves: int = information.get('total_moves')
		total_senses: int = information.get('total_senses')
		success: int = information.get('success')
		failure: int = information.get('failure')
		data: dict = {'length': length,
									'reward': reward,
									'alpha': alpha,
									'total_moves': total_moves,
									'total_senses': total_senses,
									'success': success,
									'failure': failure}

		return data
	def should_continue(self) -> bool:
		if self.episodes:
			if self.episode_count >= self.episodes:
				return False

		return True
	def _on_step(self) -> bool:
		informations: list = self.locals.get('infos')

		for information in informations:
			episode: dict = information.get('episode')

			if episode:
				data: dict = self.get_data(information)

				self.log_episode(data = data)

		should_continue: bool = self.should_continue()

		return should_continue
	def _on_training_end(self) -> None:
		if self.progress_bar:
			self.progress_bar.close()

			self.progress_bar = None
class StepCallback(BaseCallback):
	def __init__(self,
							 data_directory: str = None,
							 first_episode_logged: bool = False,
							 verbose: int = None) -> None:
		if data_directory is None:
			raise ValueError("Data directory not specified.")
		if first_episode_logged is None:
			raise ValueError("First episode logged not specified.")
		if verbose is None:
			raise ValueError("Verbose level not specified.")

		super().__init__(verbose = verbose)

		self.data_directory: str = data_directory
		self.episode_count: int = 0
		self.step_count: int = 0
		self.episode_steps: list = []
		self.first_episode_logged: bool = first_episode_logged
	def append_step(self,
                  data: dict = None) -> None:
		if data is None:
			raise ValueError("Data not specified.")

		self.step_count += 1
		action: int = data.get('action')
		reward: float = data.get('reward')
		alpha: float = data.get('alpha')
		total_moves: int = data.get('total_moves')
		total_senses: int = data.get('total_senses')
		total_actions: float = total_moves + total_senses
		move_ratio: float = float(total_moves) / float(total_actions)
		sense_ratio: float = float(total_senses) / float(total_actions)
		success: int = data.get('success')
		failure: int = data.get('failure')
		row: list = [self.step_count,
								 action,
								 reward,
								 alpha,
								 total_moves,
								 move_ratio,
								 total_senses,
								 sense_ratio,
								 success,
								 failure]

		self.episode_steps.append(row)
	def write_steps(self,
									episode_type: str = None) -> None:
		if episode_type is None:
			raise ValueError("Episode type not specified.")

		episode_step_path = os.path.join(self.data_directory,
										 								 f'{episode_type}_episode.csv')

		with open(episode_step_path,
              "w",
              encoding = "utf-8",
              newline = "") as file:
			writer: csv.writer = csv.writer(file)

			writer.writerow(["step",
											 "action",
											 "reward",
											 "alpha",
											 "total_moves",
											 "move_ratio",
											 "total_senses",
											 "sense_ratio",
											 "success",
											 "failure"])
			writer.writerows(self.episode_steps)

		self.step_count = 0
		self.episode_steps = []
	def _on_step(self) -> bool:
		informations: list = self.locals.get('infos')
		actions: list = self.locals.get('actions')
		rewards: list = self.locals.get('rewards')
		information: dict = informations[0]
		action: float = actions[0]
		reward: float = rewards[0]
		episode: dict = information.get('episode')
		alpha: float = information.get('alpha')
		total_moves: int = information.get('total_moves')
		total_senses: int = information.get('total_senses')
		success: int = information.get('success')
		failure: int = information.get('failure')
		data: dict = {'action': action,
									'reward': reward,
									'alpha': alpha,
									'total_moves': total_moves,
									'total_senses': total_senses,
									'success': success,
									'failure': failure}

		self.append_step(data = data)

		if episode:
			self.episode_count += 1

			if self.first_episode_logged:
				episode_type: str = 'last'
			else:
				episode_type: str = 'first'
				self.first_episode_logged = True

			self.write_steps(episode_type = episode_type)

		return True
class CleanupCheckpointCallback(BaseCallback):
	def __init__(self,
               checkpoint_directory: str = None,
               verbose: int = 0) -> None:
		if checkpoint_directory is None:
			raise ValueError("Checkpoint directory not specified.")
		if verbose is None:
			raise ValueError("Verbose level not specified.")

		super().__init__(verbose = verbose)

		self.checkpoint_directory: str = checkpoint_directory
	def _on_step(self) -> bool:
		return True
	def _on_training_end(self) -> None:
		cleanup_checkpoints(checkpoint_directory = self.checkpoint_directory)
