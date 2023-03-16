import pandas as pd
import os
import argparse

Episode_num = 400

def get_evaluation_data(all_data):
    data = pd.concat(all_data, axis=0, join='inner', ignore_index=True)
    data['Crashed'] = data['Crashed'].astype(float)
    data['Costs'] = data['Costs'].astype(float)
    data['Success'] = data['Success'].astype(float)
    data['Length'] = data['Length'].astype(float)
    agent_name = data['Agent'].unique()
    for index, name in enumerate(agent_name):
        # collision counter
        collision = data.loc[(data['Agent'] == name) & (data['Crashed'] == 1) ]
        collision_num = collision['Crashed'].sum()
        # cost counter
        # cost_data = data.loc[(data['Agent'] == name) & (data['Costs'] > 0.5) & (data['Success'] == 1)]
        cost_data = data.loc[(data['Agent'] == name)]
        average_cost = cost_data['Costs'].mean()
        # success counter
        success_data = data.loc[(data['Agent'] == name) & (data['Costs'] <= 0.5)]
        success_num = success_data['Success'].sum()
        # Eplen
        len_data = data.loc[(data['Agent'] == name) & (data['Success'] == 1)]
        total_inter_num = len_data['Length'].sum()
        average_time = len_data['Length'].mean() * 0.1 * 5

        print('-------------')
        print('Agent:', name)
        print('collision counter', collision_num)
        print('interact num', total_inter_num)
        print('collision_ratio: {:.4f}'.format(collision_num / Episode_num))
        print('average_cost:',average_cost)
        print('success_ratio:{:.4f}'.format(success_num / Episode_num))
        print('average_time:',average_time)

def get_all_data(file_path):
    all_data = []
    for root, dirs, files in os.walk(file_path):
        if 'process.txt' in files:
            data_path = os.path.join(root, 'process.txt')
            # data_path = os.path.join(root, 'progress.txt')
            exp_data = pd.read_table(data_path)
            exp_data = exp_data[3:]
            all_data.append(exp_data)

    return all_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='eval_result/original/low')
    # parser.add_argument("--file_path", type=str, default='eval_result/ppo/mixed')
    args = parser.parse_args()

    all_data = get_all_data(args.file_path)
    get_evaluation_data(all_data)