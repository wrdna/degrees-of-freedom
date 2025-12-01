"""
Cross-Initialization Lottery Subspace Transfer Experiment

Tests whether lottery subspaces are architecture-intrinsic or initialization-specific.
Trains networks with different seeds using a lottery subspace computed from a donor seed.

Usage:
    # First, run standard lottery_subspace.py to generate a donor subspace:
    python lottery_subspace.py --model TinyCNN --dataset MNIST --points_to_collect 1
    
    # Then run this script to test transfer from that subspace:
    python lottery_subspace_transfer.py --model TinyCNN --dataset MNIST \
        --transfer_from ../lottery-subspace-data/artifact_lottery_subspace_TinyCNN_MNIST_XXXX_grad00.pkl \
        --points_to_collect 5
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import jax.profiler

# Put the imports from the sub-functions here
from architectures import SimpleCNN, KerasResNets, WideResnet
from generate_data import setupMNIST, setupFashionMNIST, setupCIFAR10, setupCIFAR100, setupSVHN
from training_utils import generate_projection, flatten_leaves, theta_to_paramstree, sparse_theta_to_paramstree
from data_utils import save_obj, load_obj, sizeof_fmt
from logging_tools import loggingSetup, gitstatus, envstatus, rnginit

import matplotlib.pyplot as plt
import numpy as onp 
import jax.numpy as jnp 
import math
import jax
import flax
import tensorflow as tf

import time
from scipy import sparse



# Sub-functions for training 
@jax.vmap
def cross_entropy_loss(logits, label):
	return -logits[label]

@jax.jit
def normal_loss(params, batch):
	logits = jax.nn.log_softmax(net.call(params, batch['image']))
	loss = jnp.mean(cross_entropy_loss(logits, batch['label']))
	return loss

@jax.jit
def normal_accuracy(params,batch):
	logits = jax.nn.log_softmax(net.call(params, batch['image']))
	return jnp.mean(jnp.argmax(logits, -1) == batch['label'])

@jax.jit
def lr_schedule(it):
	its_in_epoch = int(len(x_train) / 128.0)

	it_thresholds = jnp.array([10,20,30,40,50,60,70,80])*its_in_epoch
	lrs = jnp.array([1.6e-3,
				1.6e-3/2,
				1.6e-3/4,
				1.6e-3/8,
				1.6e-3/16,
				1.6e-3/32,
				1.6e-3/64,
				1.6e-3/128]
				)

	return lrs[jnp.sum(it>it_thresholds)]

# Just a version of the loss function that
# takes the model rather than the params
@jax.jit
def normal_loss_opt(model, batch):
	logits = jax.nn.log_softmax(model(batch['image']))
	loss = jnp.mean(cross_entropy_loss(logits, batch['label']))
	return loss


# Set up the arguements for main
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3, help='training epochs per run')
parser.add_argument('--points_to_collect', type=int, default=2, help = 'runs per dimension (different seeds)')
parser.add_argument('--lr', type=float, default=5e-2, help = 'learning rate')
parser.add_argument('--model', type=str, default='TinyCNN', help = 'model to train')
parser.add_argument('--dataset', type=str, default='MNIST', help = 'dataset for training')
parser.add_argument('--opt_alg', type=str, default='Adam', help = 'algorithm for optimization in the plane')
parser.add_argument('--ds_to_explore', nargs='+', default=[2**x for x in range(10)], help = 'dimensions to explore')
parser.add_argument('--nnz', type=int, default=2**13, help = 'number of non-zeros in sparse projection matrix')
parser.add_argument('--traj_steps', type=int, default=5, help = 'Spacing in saving out gradient direction')
parser.add_argument('--block_start', type=int, default=0, help = 'how many points have been run before for this experiment')
parser.add_argument('--jit_grad', default=False, action='store_true', help = 'jit the gradient function for smaller projection matrices')

# NEW: Transfer experiment arguments
parser.add_argument('--transfer_from', type=str, default=None, 
	help='Path to donor lottery subspace .pkl file (e.g., artifact_lottery_subspace_..._grad00.pkl). If not provided, uses own subspace (baseline).')
parser.add_argument('--seed_offset', type=int, default=100, 
	help='Offset added to point_id for seed generation to ensure different seeds than donor')


def main(args):
	## Load in args which set parameters for runs
	epochs = args.epochs
	points_to_collect = args.points_to_collect #number of repetitions per d
	lr = args.lr
	model = args.model
	dataset = args.dataset
	opt_alg = args.opt_alg	
	ds_to_explore = [int(d_num) for d_num in args.ds_to_explore]
	nnz = args.nnz
	traj_steps = args.traj_steps
	block_start = args.block_start
	use_sparse = False
	jit_grad = args.jit_grad
	
	# NEW: Transfer settings
	transfer_from = args.transfer_from
	seed_offset = args.seed_offset
	is_transfer = transfer_from is not None

	ds_max = max(ds_to_explore)
	max_it = ds_max*traj_steps

	# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
	# it unavailable to JAX.
	tf.config.experimental.set_visible_devices([], "GPU")

	## Logging 
	do_log = True
	do_gitchecks = True
	do_envchecks = True

	log_dir = '../cross-init-lottery-subspace-data'
	
	# NEW: Different naming for transfer experiments
	if is_transfer:
		param_str = '%s_%s_transfer' % (model, dataset)
	else:
		param_str = '%s_%s_baseline' % (model, dataset)

	logger = logging.getLogger("my logger")
	scriptname = os.path.basename(__file__).rstrip('.py')
	aname, _ = loggingSetup(logger, scriptname, log_dir, do_log=do_log, param_str = param_str)
	result_file = '%s_results' % (aname)

	if do_gitchecks:
		gitstatus(logger)

	if do_envchecks:
		envstatus(logger, use_gpu = True)

	# Start log with experimental parameters
	logger.info('\n ---Code Output---\n')
	logger.info('\n')
	
	# NEW: Log transfer experiment info
	if is_transfer:
		logger.info('[Cross-Init Transfer] Training with TRANSFERRED lottery subspace: \n')
		logger.info('Donor subspace file: %s \n' % transfer_from)
	else:
		logger.info('[Cross-Init Baseline] Training with OWN lottery subspace: \n')
	
	logger.info('\n')
	logger.info('Dimensions to Explore: %s \n' % str(ds_to_explore))
	logger.info('Model: %s \n' % (model))
	logger.info('Dataset: %s \n' % (dataset))
	logger.info('Optimization Algorithm: %s with learning rate %.2e \n' % (opt_alg, lr))
	logger.info('Seed offset: %i (seeds will be %i, %i, ...)\n' % (seed_offset, seed_offset, seed_offset+1))
	logger.info('Collect %i points for each dimension. \n' % (points_to_collect))
	logger.info('Run optimization for %i epochs. \n' % (epochs))
	logger.info('\n')

	# NEW: Load donor subspace if transfer mode
	if is_transfer:
		logger.info('Loading donor subspace from: %s\n' % transfer_from)
		donor_data = load_obj(transfer_from.replace('.pkl', ''))
		U_donor = donor_data["U"]
		sigma_donor = donor_data["sigma"]
		logger.info('Donor subspace shape: %s\n' % str(U_donor.shape))
		logger.info('Donor top 10 singular values: %s\n' % str(sigma_donor[:10]))

	## Setup data
	if (dataset == 'MNIST'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupMNIST()
		input_shape = (1, 28, 28, 1)
	elif (dataset == 'fashionMNIST'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupFashionMNIST()
		input_shape = (1, 28, 28, 1)
	elif (dataset == 'SVHN'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupSVHN()
		input_shape = (1, 32, 32, 3)
	elif (dataset == 'cifar10'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupCIFAR10()
		input_shape = (1, 32, 32, 3)
	elif (dataset == 'cifar100'):
		x_train, full_train_dict, train_ds, test_ds, classes = setupCIFAR100()
		input_shape = (1, 32, 32, 3)
	else:
		logging.error('Dataset not recognized \n')

	test_ds_normalized = dict(test_ds)

	## Initialize model
	global net
	if (model == 'TinyCNN'):
		net = SimpleCNN.partial(
			channels = [16,32],
			classes = classes,
			)
	elif (model == 'SmallCNN'):
		net = SimpleCNN.partial(
			channels = [32,64,64],
			classes = classes,
			)
	elif (model == 'MediumCNN'):
		net = SimpleCNN.partial(
			channels = [32,64,64,128],
			classes = classes,
			)
	elif (model == 'ResNet_BNotf'):
		net = KerasResNets.partial(
			num_classes = classes,
			use_batch_norm = True,
		)
	elif (model == 'WideResNet'):
		net = WideResnet.partial(
			blocks_per_group=2,
			channel_multiplier=4,
			num_outputs=100,
			dropout_rate=0.0
		)
	else:
		logger.error('Model type not recognized\n')


	out = {
		"model": model,
		"dataset": dataset,
		"epochs": epochs,
		"points_to_collect": points_to_collect,
		"ds_to_explore": ds_to_explore,
		"traj_steps": traj_steps,
		"nnz": nnz,
		"full_d": '',
		"is_transfer": is_transfer,  # NEW: Track if this is a transfer experiment
		"transfer_from": transfer_from,  # NEW: Track donor file
		"seed_offset": seed_offset,  # NEW: Track seed offset
		"data": {
			"d": [],
			"point_id": [],
			"seed": [],  # NEW: Track actual seed used
			"it": [],
			"abs_theta": [],
			"train_loss": [],
			"train_acc": [],
			"full_train_loss": [],
			"full_train_acc": [],
			"best_train_acc": [],
			"test_loss": [],
			"test_acc": [],
			"best_test_acc": [],
			"nnz": [],
			"avg_grad_time": [],
			"avg_proj_time": [],
			"epoch_times": []
		}
	}

	time_per_run = onp.zeros((len(ds_to_explore), points_to_collect, epochs))

	loss_grad_full = jax.jit(jax.grad(
		lambda model, batch: normal_loss_opt(
			model,batch
		)
	))

	# Loop over runs for each seed
	for point_id in range(points_to_collect):

		# NEW: Use offset seed to ensure different from donor
		actual_seed = point_id + seed_offset + 12574
		
		out_grad = {}

		# Initialize with THIS seed's parameters (not donor's)
		_, initial_params = net.init_by_shape(jax.random.PRNGKey(actual_seed),[(input_shape, jnp.float32)])
		model = flax.nn.Model(net, initial_params)
		optimizer = flax.optim.Momentum(learning_rate=lr).create(model)

		D = jnp.sum(jnp.asarray([onp.prod(x.shape) for x in jax.tree_flatten(initial_params)[0]]))

		logger.info('\n' + '='*95 + '\n')
		logger.info('Point %i | Seed %i | D=%i\n' % (point_id, actual_seed, D))
		
		if is_transfer:
			# USE DONOR'S SUBSPACE
			U = U_donor
			sigma = sigma_donor
			logger.info('Using TRANSFERRED subspace from donor\n')
		else:
			# COMPUTE OWN SUBSPACE (baseline comparison)
			logger.info('Computing OWN subspace (baseline)...\n')
			
			smooth_grad = onp.zeros(shape = (ds_max, D))
			leaves0,treedef = jax.tree_flatten(initial_params)
			vec0,shapes_list = flatten_leaves(leaves0)
			params_old = vec0

			total_it = -1
			row = 0

			for batch in train_ds:
				total_it = total_it + 1
				if total_it > max_it:
					break

				optimizer = optimizer.apply_gradient(loss_grad_full(optimizer.target, batch))

				if (total_it % traj_steps == 0) and (total_it > 0):
					params_now = optimizer.target.params
					leaves0,treedef = jax.tree_flatten(params_now)
					vec0,shapes_list = flatten_leaves(leaves0)
					smooth_grad[row, :] = vec0 - params_old
					params_old = vec0
					row = row + 1

			U, sigma, _ = onp.linalg.svd(onp.transpose(smooth_grad), full_matrices=False)
			
			# Reset optimizer to initial params for fair comparison
			_, initial_params = net.init_by_shape(jax.random.PRNGKey(actual_seed),[(input_shape, jnp.float32)])
			
		# Save subspace info
		out_grad["U"] = U
		out_grad["sigma"] = sigma
		out_grad["initial_params"] = initial_params
		out_grad["seed"] = actual_seed
		out_grad["is_transfer"] = is_transfer

		if point_id < 10:
			grad_file = '%s_grad0%i' % (aname, point_id)
		else:
			grad_file = '%s_grad%i' % (aname, point_id)
		save_obj(out_grad, grad_file, log_dir + '/') 

		# Loop over dimensions to explore
		for d_num, d in enumerate(ds_to_explore):

			logger.info('\n'+'-'*95+'\n')
			logger.info("Run Number "+str(point_id)+" | Seed "+str(actual_seed)+'\n')
			logger.info("Number of params = "+str(D)+"   subspace d="+str(d)+'\n')

			# Projection plane from lottery subspace
			M = onp.transpose(U[:, 0:d])
			M_unit = M / onp.linalg.norm(M,axis=1,keepdims=True)

			# Use THIS seed's initial params as the center point
			leaves0,treedef = jax.tree_flatten(initial_params)
			vec0,shapes_list = flatten_leaves(leaves0)

			if jit_grad:
				loss_grad_wrt_theta = jax.jit(jax.grad(
					lambda theta_now, batch: normal_loss(
						theta_to_paramstree(theta_now,M_unit,vec0,treedef,shapes_list), batch
					)
				))
			else:
				loss_grad_wrt_theta = jax.grad(
					lambda theta_now, batch: normal_loss(
						theta_to_paramstree(theta_now,M_unit,vec0,treedef,shapes_list), batch
					)
				)

			# Start at the initial params (vec0), not the global origin
			theta = jnp.zeros((1,d)) 

			# Parameters and aux variables for Adam
			beta_1=0.9
			beta_2=0.999
			epsilon=1e-07

			mass = jnp.zeros((1, d))
			velocity = jnp.zeros((1, d))

			total_it = -1
			best_train_acc = 0
			best_test_acc = 0
			
			grad_ts = []
			proj_ts = []

			## Train the model
			for batch in train_ds:

				total_it += 1

				if total_it / (len(x_train)/128.0) > epochs:
					break

				e_float = total_it / (len(x_train)/128.0)

				grad_t1 = time.time()
				g_theta = loss_grad_wrt_theta(theta,batch)
				grad_t2 = time.time()
				grad_ts.append(grad_t2 - grad_t1)

				if (opt_alg == 'Adam'):
					mass = beta_1 * mass + (1.0 - beta_1) * g_theta
					velocity = beta_2 * velocity + (1.0 - beta_2) * (g_theta**2.0)
					hat_mass = mass / (1.0-beta_1)
					hat_velocity = velocity / (1.0-beta_2)
					theta = theta - lr / (jnp.sqrt(hat_velocity) + epsilon) * hat_mass
				else:
					theta = theta - lr*g_theta

				proj_t1 = time.time()
				params_now = theta_to_paramstree(theta,M_unit,vec0,treedef,shapes_list)
				proj_t2 = time.time()
				proj_ts.append(proj_t2 - proj_t1)

				loss_out = normal_loss(params_now,batch)
				accuracy_out = normal_accuracy(params_now,batch)

				if total_it % 50 == 0 and total_it != 0:
					logger.info('{:10}{:10}{:15}{:15}{:15}{:15}{:15}'.format(str(round(e_float, 3)),str(total_it),str(onp.linalg.norm(theta)),str(loss_out),str(accuracy_out),'-','-')+'\n')

				if (total_it % int(len(x_train)/128.0)) in [0]:
					test_loss_out = normal_loss(params_now,test_ds_normalized)
					test_accuracy_out = normal_accuracy(params_now,test_ds_normalized)
					full_loss_out = normal_loss(params_now,full_train_dict)
					full_accuracy_out = normal_accuracy(params_now,full_train_dict)

					if test_accuracy_out > best_test_acc:
						best_test_acc = test_accuracy_out
					if full_accuracy_out > best_train_acc:
						best_train_acc = full_accuracy_out

					if total_it > 0:
						t2 = time.time()
						time_per_run[d_num, point_id, int(total_it / int(len(x_train)/128.0))-1] = t2 - t1
					t1 = time.time()
					
					logger.info('{:10}{:10}{:15}{:15}{:15}{:15}{:15}'.format('epoch','iter','|theta|', 'train loss', 'train acc', 'test loss', 'test acc')+'\n')
					logger.info('{:10}{:10}{:15}{:15}{:15}{:15}{:15}'.format(str(round(e_float, 3)),str(total_it),str(onp.linalg.norm(theta)),str(full_loss_out),str(full_accuracy_out),str(test_loss_out),str(test_accuracy_out))+'\n')


			avg_grad_time = onp.mean(grad_ts)
			avg_proj_time = onp.mean(proj_ts)

			logger.info('\nTotal time:                     ' + str(sum(time_per_run[d_num, point_id])) +'\n')
			logger.info('Avg time to compute gradient:   ' + str(avg_grad_time)+'\n')
			logger.info('Avg time to project theta:      ' + str(avg_proj_time)+'\n')

			# Data out
			out["full_d"] = D
			out["data"]["d"].append(d)
			out["data"]["point_id"].append(point_id)
			out["data"]["seed"].append(actual_seed)  # NEW
			out["data"]["it"].append(str(total_it))
			out["data"]["abs_theta"].append(str(onp.linalg.norm(theta)))
			out["data"]["train_loss"].append(str(loss_out))
			out["data"]["train_acc"].append(str(accuracy_out))
			out["data"]["full_train_loss"].append(str(full_loss_out))
			out["data"]["full_train_acc"].append(str(full_accuracy_out))
			out["data"]["best_train_acc"].append(str(best_train_acc))
			out["data"]["test_loss"].append(str(test_loss_out))
			out["data"]["test_acc"].append(str(test_accuracy_out))
			out["data"]["best_test_acc"].append(str(best_test_acc))
			out["data"]["nnz"].append(nnz)
			out["data"]["avg_grad_time"].append(avg_grad_time)
			out["data"]["avg_proj_time"].append(avg_proj_time)
			out["data"]["epoch_times"].append(time_per_run[d_num, point_id])

		# Write data after each seed
		save_obj(out, result_file, log_dir + '/') 

	# Final summary
	logger.info('\n' + '='*95 + '\n')
	logger.info('EXPERIMENT COMPLETE\n')
	logger.info('Transfer mode: %s\n' % is_transfer)
	if is_transfer:
		logger.info('Donor file: %s\n' % transfer_from)
	logger.info('Seeds used: %s\n' % str([seed_offset + i + 12574 for i in range(points_to_collect)]))
	logger.info('Results saved to: %s\n' % result_file)


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)

