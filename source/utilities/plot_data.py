import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

def get_data(data_directory: str = None) -> dict[float,
																								 pd.DataFrame]:
	if data_directory is None:
		raise ValueError("Data directory not specified.")

	data: dict[float,
						 pd.DataFrame] = {}
	alpha_directories: list[str] = glob.glob(os.path.join(data_directory,
                                                   			'alpha_*'))

	for alpha_directory in alpha_directories:
		folder_name: str = os.path.basename(alpha_directory)
		alpha_str: str = folder_name.split('_')[1]
		alpha: float = float(alpha_str)
		csv_files: list[str] = glob.glob(os.path.join(alpha_directory,
                                                  '*.csv'))

		for csv_file in csv_files:
			if 'first_episode' not in csv_file and 'last_episode' not in csv_file:
				data[alpha] = pd.read_csv(csv_file)

				break

	return data
def get_cumulative_rewards(episode_data: pd.DataFrame = None) -> list[float]:
	if episode_data is None:
		raise ValueError("Episode data not specified.")

	cumulative_rewards: list[float] = []
	total_reward: float = 0.0

	for index in range(len(episode_data)):
		total_reward += episode_data['reward'].iloc[index]

		cumulative_rewards.append(total_reward)

	return cumulative_rewards
def get_cumulative_values(episode_data: pd.DataFrame = None,
													metric_values: str = None,
													reduction_values: str = None) -> list[float]:
	if episode_data is None:
		raise ValueError("Episode data not specified.")
	if metric_values is None:
		raise ValueError("Metric values not specified.")
	if reduction_values is None:
		raise ValueError("Reduction values not specified.")

	starting_value: float = episode_data[metric_values].iloc[0]
	cumulative_values: list[float] = [starting_value]

	for index in range(1,
                     len(episode_data)):
		next_value: float = cumulative_values[-1] - episode_data[reduction_values].iloc[index]

		cumulative_values.append(next_value)

	return cumulative_values
def plot_episode(episode_data: pd.DataFrame = None,
								 episode_title: str = None,
								 alpha: float = None,
								 episode_plot_path: str = None) -> None:
	if episode_data is None:
		raise ValueError("Episode data not specified.")
	if episode_title is None:
		raise ValueError("Episode title not specified.")
	if alpha is None:
		raise ValueError("Alpha not specified.")
	if episode_plot_path is None:
		raise ValueError("Episode plot path not specified.")

	(_,
   axes) = plt.subplots(2,
                        2,
                        figsize = (16,
                                   12))
	step: pd.Series = episode_data['step']
	cumulative_rewards: list[float] = get_cumulative_rewards(episode_data = episode_data)
	move_ratio: pd.Series = episode_data['move_ratio']
	sense_ratio: pd.Series = episode_data['sense_ratio']
	cumulative_distances: list[float] = get_cumulative_values(episode_data = episode_data,
																														metric_values = 'distance_to_maximum_belief',
																														reduction_values = 'distance_to_maximum_belief_reduction')
	cumulative_entropies: list[float] = get_cumulative_values(episode_data = episode_data,
																														metric_values = 'belief_shannon_entropy',
																														reduction_values = 'belief_shannon_entropy_reduction')

	axes[0,
       0].plot(step,
               cumulative_rewards,
							 linewidth = 2,
               color = 'blue',
               marker = 'o',
               markersize = 4)
	axes[0,
       0].set_title(f'Cumulative Reward over Step (alpha = {alpha:.2f}, {episode_title})',
										fontsize = 13,
                    fontweight = 'bold')
	axes[0,
       0].set_xlabel('Step',
                     fontsize = 11)
	axes[0,
       0].set_ylabel('Cumulative Reward',
                     fontsize = 11)
	axes[0,
       0].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[0,
       1].plot(step,
               move_ratio,
							 label = 'Move',
               linewidth = 2,
               color = 'skyblue',
               marker = 's',
               markersize = 4)
	axes[0,
       1].plot(step,
               sense_ratio,
							 label = 'Sense',
							 linewidth = 2,
							 color = 'orange',
							 marker = 'D',
							 markersize = 4)
	axes[0,
       1].set_title(f'Move and Sense Ratio over Step (alpha = {alpha:.2f}, {episode_title})',
										fontsize = 13,
                    fontweight = 'bold')
	axes[0,
       1].set_xlabel('Step',
                     fontsize = 11)
	axes[0,
       1].set_ylabel('Ratio',
                     fontsize = 11)
	axes[0,
       1].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[0,
       1].set_ylim([0, 
                    1.05])
	axes[0,
       1].legend(fontsize = 11)
	axes[1,
       0].plot(step,
							 cumulative_distances,
							 linewidth = 2,
							 color = 'red',
							 marker = 'o',
							 markersize = 4)
	axes[1,
       0].set_title(f'Distance to Maximum Belief over Step (alpha = {alpha:.2f}, {episode_title})',
										fontsize = 13,
										fontweight = 'bold')
	axes[1,
       0].set_xlabel('Step',
                     fontsize = 11)
	axes[1,
       0].set_ylabel('Distance to Maximum Belief',
                     fontsize = 11)
	axes[1,
       0].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[1,
       1].plot(step,
               cumulative_entropies,
							 linewidth = 2,
               color = 'purple',
               marker = 'o',
               markersize = 4)
	axes[1,
       1].set_title(f'Belief Shannon Entropy over Step (alpha = {alpha:.2f}, {episode_title})',
										fontsize = 13,
                    fontweight = 'bold')
	axes[1,
       1].set_xlabel('Step',
                     fontsize = 11)
	axes[1,
       1].set_ylabel('Belief Shannon Entropy',
                     fontsize = 11)
	axes[1,
       1].grid(True,
               linestyle = '--',
               alpha = 0.6)

	plt.tight_layout()
	plt.savefig(episode_plot_path,
              dpi = 300,
              bbox_inches = 'tight')
	plt.close()
def plot_alpha_episodes(alpha_data_directory: str = None,
												alpha_plots_directory: str = None,
												alpha: float = None) -> None:
	if alpha_data_directory is None:
		raise ValueError("Alpha data directory not specified.")
	if alpha_plots_directory is None:
		raise ValueError("Alpha plots directory not specified.")
	if alpha is None:
		raise ValueError("Alpha not specified.")

	first_episode_path: str = os.path.join(alpha_data_directory,
                                         'first_episode.csv')
	first_episode_plot_path: str = os.path.join(alpha_plots_directory,
																				'first_episode.png')
	last_episode_path: str = os.path.join(alpha_data_directory,
                                        'last_episode.csv')
	last_episode_plot_path: str = os.path.join(alpha_plots_directory,
																				'last_episode.png')
	first_episode_data: pd.DataFrame = pd.read_csv(first_episode_path)
	last_episode_data: pd.DataFrame = pd.read_csv(last_episode_path)

	plot_episode(episode_data = first_episode_data,
							 episode_title = 'First Episode',
							 alpha = alpha,
							 episode_plot_path = first_episode_plot_path)
	plot_episode(episode_data = last_episode_data,
							 episode_title = 'Last Episode',
							 alpha = alpha,
							 episode_plot_path = last_episode_plot_path)
def get_summary_data(data: dict[float,
                                pd.DataFrame] = None,
										 summary_data_path: str = None) -> pd.DataFrame:
	if data is None:
		raise ValueError("Data not specified.")
	if summary_data_path is None:
		raise ValueError("Summary data path not specified.")

	rows: list[dict] = []

	for alpha in sorted(data.keys()):
		alpha_data: pd.DataFrame = data[alpha]
		average_length: float = alpha_data['length'].mean()
		average_reward: float = alpha_data['reward'].mean()
		average_move_ratio: float = alpha_data['move_ratio'].mean()
		average_sense_ratio: float = alpha_data['sense_ratio'].mean()
		final_success_rate: float = alpha_data['success_rate'].iloc[-1]
		final_failure_rate: float = alpha_data['failure_rate'].iloc[-1]

		rows.append({'alpha': alpha,
								 'average_length': average_length,
								 'average_reward': average_reward,
								 'average_move_ratio': average_move_ratio,
								 'average_sense_ratio': average_sense_ratio,
								 'final_success_rate': final_success_rate,
								 'final_failure_rate': final_failure_rate})

	summary_data: pd.DataFrame = pd.DataFrame(rows)

	summary_data.to_csv(summary_data_path,
                      index = False)

	return summary_data
def plot_summary_data(summary_data: pd.DataFrame = None,
											summary_plot_path: str = None) -> None:
	if summary_data is None:
		raise ValueError("Summary dataframe not specified.")
	if summary_plot_path is None:
		raise ValueError("Output path not specified.")

	(_,
   axes) = plt.subplots(2,
                        2,
                        figsize = (16,
                                   12))
	alpha: pd.Series = summary_data['alpha']
	average_length: pd.Series = summary_data['average_length']
	average_reward: pd.Series = summary_data['average_reward']
	average_move_ratio: pd.Series = summary_data['average_move_ratio']
	average_sense_ratio: pd.Series = summary_data['average_sense_ratio']
	final_success_rate: pd.Series = summary_data['final_success_rate']
	final_failure_rate: pd.Series = summary_data['final_failure_rate']

	axes[0,
       0].plot(alpha,
               average_length,
							 marker = 'o',
               linewidth = 2,
               markersize = 8,
               color = 'purple')
	axes[0,
       0].set_title('Average Length per Alpha Value',
										fontsize = 13,
                    fontweight = 'bold')
	axes[0,
       0].set_xlabel('Alpha Value',
                     fontsize = 11)
	axes[0,
       0].set_ylabel('Average Length',
                     fontsize = 11)
	axes[0,
       0].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[0,
       1].plot(alpha,
               average_reward,
							 marker = 'o',
               linewidth = 2,
               markersize = 8,
               color = 'blue')
	axes[0,
       1].set_title('Average Reward per Alpha Value',
										fontsize = 13,
                    fontweight = 'bold')
	axes[0,
       1].set_xlabel('Alpha Value',
                     fontsize = 11)
	axes[0,
       1].set_ylabel('Average Reward',
                     fontsize = 11)
	axes[0,
       1].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[1,
       0].plot(alpha,
               average_move_ratio,
							 marker = 's',
               linewidth = 2,
               markersize = 8,
               color = 'skyblue',
               label = 'Move')
	axes[1,
       0].plot(alpha,
               average_sense_ratio,
							 marker = 'D',
               linewidth = 2,
               markersize = 8,
               color = 'orange',
               label = 'Sense')
	axes[1,
       0].set_title('Average Move and Sense Ratio per Alpha Value',
										fontsize = 13,
                    fontweight = 'bold')
	axes[1,
       0].set_xlabel('Alpha Value',
                     fontsize = 11)
	axes[1,
       0].set_ylabel('Ratio',
                     fontsize = 11)
	axes[1,
       0].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[1,
       0].set_ylim([0,
                    1.05])
	axes[1,
       0].legend(fontsize = 11)
	axes[1,
       1].plot(alpha,
               final_success_rate,
							 marker = '^',
               linewidth = 2,
               markersize = 8,
               color = 'green',
               label = 'Success')
	axes[1,
       1].plot(alpha,
               final_failure_rate,
							 marker = 'v',
               linewidth = 2,
               markersize = 8,
               color = 'red',
               linestyle = '--',
               label = 'Failure')
	axes[1,
       1].set_title('Final Success and Failure Rate per Alpha Value',
										fontsize = 13,
                    fontweight = 'bold')
	axes[1,
       1].set_xlabel('Alpha Value',
                     fontsize = 11)
	axes[1,
       1].set_ylabel('Rate',
                     fontsize = 11)
	axes[1,
       1].grid(True,
               linestyle = '--',
               alpha = 0.6)
	axes[1,
       1].set_ylim([0,
                    1.05])
	axes[1,
       1].legend(fontsize = 11)

	for (row,
       column) in zip(alpha, average_length):
		axes[0,
         0].annotate(f'{column:.1f}',
                     (row,
                      column),
                     textcoords = 'offset points',
										 xytext = (0,
                               10),
                     ha = 'center',
                     fontsize = 9)
	for (row,
			 column) in zip(alpha,
                      average_reward):
		axes[0,
         1].annotate(f'{column:.1f}',
                     (row,
                      column),
                     textcoords = 'offset points',
										 xytext = (0,
                               10),
                     ha = 'center',
                     fontsize = 9)
	for (row,
			 column) in zip(alpha,
                      average_move_ratio):
		axes[1,
         0].annotate(f'{column:.2f}',
                     (row,
											column),
										 textcoords = 'offset points',
										 xytext = (0,
                               -15),
           					 ha = 'center',
                 		 fontsize = 8,
                     color = 'skyblue')
	for (row,
			 column) in zip(alpha,
                      average_sense_ratio):
		axes[1,
         0].annotate(f'{column:.2f}',
                     (row,
                      column),
                     textcoords = 'offset points',
										 xytext = (0,
                               10),
                     ha = 'center',
                     fontsize = 8,
                     color = 'orange')
	for (row,
			 column) in zip(alpha,
                      final_success_rate):
		axes[1,
         1].annotate(f'{column:.2f}',
                     (row,
                      column),
                     textcoords = 'offset points',
										 xytext = (0,
                               -15),
                     ha = 'center',
                     fontsize = 8,
                     color = 'green')
	for (row,
			 column) in zip(alpha,
                      final_failure_rate):
		axes[1,
         1].annotate(f'{column:.2f}',
                     (row,
											column),
										 textcoords = 'offset points',
										 xytext = (0,
                               10),
           					 ha = 'center',
                     fontsize = 8,
                     color = 'red')

	plt.tight_layout()
	plt.savefig(summary_plot_path,
              dpi = 300,
              bbox_inches = 'tight')
	plt.close()
def plot_data(output_directory: str = None,
              data_type: str = None,
              agent_label: str = None) -> None:
	if output_directory is None:
		raise ValueError("Output directory not specified.")
	if data_type not in ['training',
                       'evaluation']:
		raise ValueError("Data type must be 'training' or 'evaluation'.")
	if data_type == 'training':
		plots_directory: str = os.path.join(output_directory,
																			  f'{data_type}_plots')
		data_directory: str = os.path.join(output_directory,
																			 f'{data_type}_data')
	elif data_type == 'evaluation':
		if agent_label is None:
			raise ValueError("Agent label not specified.")

		plots_directory: str = os.path.join(output_directory,
																			  f'{data_type}_plots',
																			  agent_label)
		data_directory: str = os.path.join(output_directory,
																			 f'{data_type}_data',
																			 agent_label)
	else:
		raise ValueError("Data type must be 'training' or 'evaluation'.")

	summary_data_path: str = os.path.join(plots_directory,
																			  'summary.csv')
	summary_plot_path: str = os.path.join(plots_directory,
																			  'summary.png')
	data_label: str = data_type.upper()

	os.makedirs(plots_directory,
              exist_ok = True)

	print("\n" + "=" * 80)
	print(f"GENERATING {data_label} PLOTS")
	print("=" * 80)

	data: dict[float,
             pd.DataFrame] = get_data(data_directory = data_directory)

	print(f"Found {data_type} data for {len(data)} alpha values")

	for alpha in sorted(data.keys()):
		print(f"Plotting step-level episodes for alpha {alpha:.2f}...")
		
		alpha_data_directory: str = os.path.join(data_directory,
                                          	 f'alpha_{alpha:.2f}')
		alpha_plots_directory: str = os.path.join(plots_directory,
                                           		f'alpha_{alpha:.2f}')

		os.makedirs(alpha_plots_directory,
                exist_ok = True)
		plot_alpha_episodes(alpha_data_directory = alpha_data_directory,
												alpha_plots_directory = alpha_plots_directory,
												alpha = alpha)

	print("Generating summary statistics...")

	summary_data: pd.DataFrame = get_summary_data(data = data,
																								summary_data_path = summary_data_path)

	print("Creating summary comparison plot...")
	
	plot_summary_data(summary_data = summary_data,
									  summary_plot_path = summary_plot_path)

	print(f"\n{data_label.capitalize()} plots generated successfully in: {plots_directory}")
	print("=" * 80)
