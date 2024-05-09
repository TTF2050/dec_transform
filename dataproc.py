#!/bin/env python

#for each experiment
    #for each batch
        # load 13 variants
        # for each hyperparam in length, layers, heads, batch, embed, lr
            # reward mean/variance over batches at each param value
            # linear regression (is there a trend?)
            # variance over all samples
            # if variance is different, then the hyperparam is important

from collections import defaultdict
from copy import deepcopy
from os import path
import os
import re
import matplotlib.pyplot as plt


import numpy as np
from sklearn.linear_model import LinearRegression

script_dir = path.dirname(path.realpath(__file__))

batches = ('batch1', 'batch2', 'batch3')
results_dir = 'results'
experiments = ('hopper', 'halfcheetah', 'walker2d')
datasets = ('expert',)

param_defaults = {
    "seq_len": '20',
    "layers": '3',
    "heads": '1',
    "batch_size": '64',
    "embed": '128',
    "learn_rate": '1e-4'
}

param_alts = {
    "seq_len": ('10', '30'),
    "layers": ('4', '5'),
    "heads": ('2', '4'),
    "batch_size": ('32', '128'),
    "embed": ('64', '256'),
    "learn_rate": ('2e-4', '5e-5')
}

pretty_names = {
    "seq_len": "sequence length",
    "layers": "layer count",
    "heads": "head count",
    "batch_size": "batch size",
    "embed": "embedding dimension",
    "learn_rate": "learning rate"
}

exp_tgt_returns = {
    'hopper': 3600,
    'halfcheetah': 12000,
    'walker2d': 5000
}


def build_log_name(exp, dataset, params):
    file_name = f"{exp}-{dataset}-"
    file_name += f"{params['seq_len']}-"
    file_name += f"{params['layers']}-"
    file_name += f"{params['heads']}-"
    file_name += f"{params['batch_size']}-"
    file_name += f"{params['embed']}-"
    file_name += f"{params['learn_rate']}.log"
    return file_name



exp_description_pattern = re.compile(r"Starting new experiment: (\w+) (\w+)")
avg_exp_return_pattern = re.compile(r"Average return: (\d+\.\d+), std: (\d+\.\d+)")
max_exp_return_pattern = re.compile(r"Max return: (\d+\.\d+), min: (\d+\.\d+)")

dectransformer_pattern = re.compile(r"DecisionTransformer")

itr_pattern = re.compile(r"Iteration (\d+)")
train_time_pattern = re.compile(r"time/training: (\d+\.\d+)")

eval_return_mean_pattern = re.compile(r"evaluation/target_(\d+)_return_mean: (\d+\.\d+)")
eval_return_std_pattern = re.compile(r"evaluation/target_(\d+)_return_std: (\d+\.\d+)")
eval_len_mean_pattern = re.compile(r"evaluation/target_(\d+)_length_mean: (\d+\.\d+)")
eval_len_std_pattern = re.compile(r"evaluation/target_(\d+)_length_std: (\d+\.\d+)")

train_loss_mean_pattern = re.compile(r"training/train_loss_mean: (\d+\.\d+)")
train_loss_std_pattern = re.compile(r"training/train_loss_std: (\d+\.\d+)")
train_act_err_pattern = re.compile(r"training/action_error: (\d+\.\d+)")

def process_log_file(log_file):

    def load_iter(i, f):
        itr_data={}
        #discard spacer or model summary
        l = f.readline()
        match = dectransformer_pattern.match(l)
        if match is not None:
            for _ in range(36):
                l = f.readline()
                # print(f'discarding {l}')

        #iteration line
        match = itr_pattern.match(f.readline())
        assert match is not None , "file desync"
        assert match.group(1) == str(i), f'match failed: {match.group(1)} vs {str(i)}'
        #discard training time
        f.readline()
        #max return matrics (4 lines)
        match = eval_return_mean_pattern.match(f.readline())
        assert match is not None , "eval return mean fail"
        itr_data['target_return'] = int(match.group(1))
        itr_data['return_mean'] = float(match.group(2))
        
        match = eval_return_std_pattern.match(f.readline())
        assert match is not None , "eval return std fail"
        itr_data['return_std'] = float(match.group(2))

        match = eval_len_mean_pattern.match(f.readline())
        assert match is not None , "eval len mean fail"
        itr_data['length_mean'] = float(match.group(2))

        match = eval_len_std_pattern.match(f.readline())
        assert match is not None , "eval len std fail"
        itr_data['length_std'] = float(match.group(2))

        #burn the next 4
        f.readline()
        f.readline()
        f.readline()
        f.readline()

        #discard time metrics (2)
        f.readline()
        f.readline()

        #loss metrics (3)
        match = train_loss_mean_pattern.match(f.readline())
        assert match is not None , "train loss mean fail"
        itr_data['loss_mean'] = float(match.group(1))

        match = train_loss_std_pattern.match(f.readline())
        assert match is not None , "train loss std fail"
        itr_data['loss_std'] = float(match.group(1))
        
        match = train_act_err_pattern.match(f.readline())
        assert match is not None , "train action err fail"
        itr_data['action_error'] = float(match.group(1))

        return itr_data
        
    logged_values = defaultdict(list)
    with open(log_file) as f:
        
        # parse headers
        print(f.readline())        
        for _ in range(3):
            f.readline()

        match = avg_exp_return_pattern.match(f.readline())
        assert match is not None , "avg_exp_return_pattern fail"
        logged_values['avg_exp_return_mean'] = float(match.group(1))
        logged_values['avg_exp_return_std'] = float(match.group(2))

        match = max_exp_return_pattern.match(f.readline())
        assert match is not None , "max_exp_return_pattern fail"
        logged_values['max_exp_return_mean'] = float(match.group(1))
        logged_values['min_exp_return_std'] = float(match.group(2))

        f.readline()

        # expect 10 data chunks
        for i in range(10):
            itr_data = load_iter(i+1,f)
            # print(itr_data)
            for k in itr_data.keys():
                logged_values[k].append(itr_data[k])
    
    return logged_values

def collect_data(base_path):
    all_data = {}
    for exp in experiments:
        all_data[exp] = {}
        for dataset in datasets:
            all_data[exp][dataset] = {}
            for batch in batches:
                all_data[exp][dataset][batch] = {}
                batch_dir = path.join(base_path,batch)
        
                for test_param in param_defaults.keys():
                    all_data[exp][dataset][batch][test_param] = {}
                    # test_param_values = []
                    # data_sets_for_test_param = []
                    #load the prime dataset
                    params = deepcopy(param_defaults)
                    log_name = build_log_name(exp, dataset, params)
                    log_file = path.join(batch_dir,log_name)
                    log_data = process_log_file(log_file)
                    # print(log_data)
                    all_data[exp][dataset][batch][test_param][params[test_param]] = log_data
                    #load the variants
                    for alt_value in param_alts[test_param]:
                        params = deepcopy(param_defaults)
                        params[test_param] = alt_value
                        log_name = build_log_name(exp, dataset, params)
                        log_file = path.join(batch_dir,log_name)
                        log_data = process_log_file(log_file)
                        # print(log_data)
                        all_data[exp][dataset][batch][test_param][params[test_param]] = log_data
                    # print(all_data[exp][dataset][batch][test_param]['values'])
                    # print(all_data[exp][dataset][batch][test_param]['metrics'])
                
    # print(all_data)
    return all_data

def generate_results(all_data):
    # all_data[exp][dataset][batch][test_param]['values']
    # all_data[exp][dataset][batch][test_param]['metrics']
    fig, axs = plt.subplots(2,3, dpi=600)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92], h_pad=2)
    num_els = 4
    extra_iters = 2
    fig.suptitle(f'Mean normalized reward after {num_els+1} training iterations')
    with open(f'{path.join(script_dir,results_dir)}/trained_stats.txt','w') as f:
        for test_param,ax in zip(param_defaults.keys(), fig.get_axes()):
            f.write(f'PARAMETER: {test_param}\n')
            # collect all exp rewards (normalized by target return) for a given test param value
            param_values = (param_defaults[test_param], *param_alts[test_param])
            print(test_param)
            print(param_values)
            param_data = {}

            for param_val in param_values:
                param_data[param_val] = []
                for exp in experiments:
                    for batch in batches:
                        data_set = all_data[exp]['expert'][batch][test_param][param_val]
                        # print(data_set)
                        param_data[param_val] += list(map(lambda x: x[0]/x[1], zip(data_set['return_mean'][num_els:num_els+extra_iters],data_set['target_return'][num_els:num_els+extra_iters])))
            print(param_data)
            
            y = np.zeros(0)
            x = np.zeros(0)
        
            f.write('param value       mean      stddev\n')
            for param_val in param_values:
                y_chunk = np.asarray(param_data[param_val])
                y = np.concatenate([y, y_chunk])
                x = np.concatenate([x, np.ones_like(y_chunk)*float(param_val)])
                f.write(f'{param_val}    {np.mean(y_chunk):.2f}    {np.std(y_chunk):.2f}\n')
            print(x)

            ticks = list(map(float,param_values))
            ticks.sort()

            ax.scatter(x,y)
            ax.set_title(f'{pretty_names[test_param]}')
            ax.set_xticks(ticks)
            ax.ticklabel_format(style='sci',scilimits=(0,1000),axis='both')

            ax.set_ylim(0,1.1)
            
            plt.figure('single_plot')
            plt.scatter(x,y)
            plt.title(f'Mean normalized reward after {num_els+1} training iterations')
            plt.xlabel(f'{pretty_names[test_param]}')
            plt.xticks(ticks)
            plt.ticklabel_format(style='sci',scilimits=(0,1000),axis='both')
            plt.ylabel('normalized episode returns')
            plt.ylim(0,1.1)
            plt.savefig(f'{path.join(script_dir,results_dir)}/trained_{test_param}.png')
            plt.clf()
        
    fig.savefig(f'{path.join(script_dir,results_dir)}/trained_cluster.png')
    plt.close(fig)


    def do_lin_reg(x, y_raw, y_norm):
        print(f'y_raw {y_raw}')
        print(f'y_norm {y_norm}')
        norm_rets = np.asarray(list(map(lambda x: x[0]/x[1], zip(y_raw,y_norm))))
        norm_rets = np.expand_dims(norm_rets,axis=-1)
        print(f'norm_rets {norm_rets}')
        model = LinearRegression(fit_intercept=True)
        x = np.arange(x)
        x = np.expand_dims(x,axis=-1)
        print(f'x.shape {x.shape}')
        model.fit(x, norm_rets)
        print(model.coef_)
        return np.squeeze(model.coef_)

    fig, axs = plt.subplots(2,3, dpi=600)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92], h_pad=2)
    num_els = 3
    fig.suptitle(f'Mean normalized reward improvement rate (first {num_els} iterations)')
    with open(f'{path.join(script_dir,results_dir)}/converge_stats.txt','w') as f:
        for test_param,ax in zip(param_defaults.keys(), fig.get_axes()):
            f.write(f'PARAMETER: {test_param}\n')
            # collect all exp rewards (normalized by target return) for a given test param value
            param_values = (param_defaults[test_param], *param_alts[test_param])
            # print(test_param)
            # print(param_values)
            param_data = {}
            
            for param_val in param_values:
                param_data[param_val] = []
                for exp in experiments:
                    for batch in batches:
                        data_set = all_data[exp]['expert'][batch][test_param][param_val]
                        
                        param_data[param_val].append(do_lin_reg(3,data_set['return_mean'][:num_els],data_set['target_return'][:num_els]))
            print(param_data)
            
            y = np.zeros(0)
            x = np.zeros(0)
            f.write('param value       mean      stddev\n')
            for param_val in param_values:
                y_chunk = np.asarray(param_data[param_val])
                y = np.concatenate([y, y_chunk])
                x = np.concatenate([x, np.ones_like(y_chunk)*float(param_val)])
                f.write(f'{param_val}    {np.mean(y_chunk):.2f}    {np.std(y_chunk):.2f}\n')
            print(x)

            ticks = list(map(float,param_values))
            ticks.sort()
            
            ax.scatter(x,y)
            ax.set_title(f'{pretty_names[test_param]}')
            ax.set_xticks(ticks)
            ax.ticklabel_format(style='sci',scilimits=(0,1000),axis='both')

            ax.set_ylim(0,1.1)

            plt.figure('single_plot')
            plt.scatter(x,y)
            plt.title(f'mean return improvement rate (first {num_els} iterations)')
            plt.xlabel(f'{pretty_names[test_param]}')
            
            plt.xticks(ticks)
            plt.ticklabel_format(style='sci',scilimits=(0,1000),axis='both')
            plt.ylabel('normalized episode return improvement rate')
            plt.ylim(0,1.1)
            plt.savefig(f'{path.join(script_dir,results_dir)}/convergence_{test_param}.png')
            plt.close()
    
    fig.savefig(f'{path.join(script_dir,results_dir)}/convergence_cluster.png')
    plt.close(fig)

            


if __name__ == "__main__":
    
    os.makedirs(path.join(script_dir,results_dir), exist_ok=True)
    all_data = collect_data(script_dir)
    generate_results(all_data)
