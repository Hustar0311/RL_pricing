# input: parameters
# output: trainded model under "train mode", tested results under "test mode"
from sim import Environment
from center import MAXP, MINP, Platform
import sys
import numpy as np
import argparse
import json
import os
import time
import random
from economic_action import ActionSpec
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from runner import train, test


# usage example
# python main.py -d grid_small_dynamic/3/0/ --pricing_alg TD3_MLP -alr 0.00001 -clr 0.001 --n_epochs 50 -m all -n 0 --batch_size 32 --seed 5 -sa Gaussian -pd 0 -pe 0 -ac
# TODO:
# Add another algorithm for SAR

def tag_float(value):
    return str(value).replace("-", "m").replace(".", "p")

def normalize_mechanism_flags(args):
    """Resolve the legacy all-on switch into two explicit mechanism flags."""
    legacy_all_on = bool(getattr(args, "use_bounded_rationality", False))
    args.use_passenger_entry_br = bool(
        getattr(args, "use_passenger_entry_br", False) or legacy_all_on
    )
    args.use_driver_online_br = bool(
        getattr(args, "use_driver_online_br", False) or legacy_all_on
    )
    args.use_bounded_rationality = bool(
        args.use_passenger_entry_br or args.use_driver_online_br
    )
    return args

def enabled_mechanisms(args):
    mechanisms = []
    if args.use_passenger_entry_br:
        mechanisms.append("passenger_entry")
    if args.use_driver_online_br:
        mechanisms.append("driver_online")
    return mechanisms

def bounded_rationality_tag(args):
    if not args.use_bounded_rationality:
        return ""
    parts = ["BR"]
    if args.use_passenger_entry_br:
        parts.extend([
            "passenger_entry",
            "bp{}".format(tag_float(args.beta_passenger)),
            "pu{}".format(tag_float(args.passenger_base_utility)),
            "pp{}".format(tag_float(args.passenger_price_sensitivity)),
            "pw{}".format(tag_float(args.passenger_wait_sensitivity)),
        ])
    if args.use_driver_online_br:
        parts.extend([
            "driver_online",
            "bd{}".format(tag_float(args.beta_driver)),
        ])
    return "_" + "_".join(parts)

def _read_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return {}

def write_metadata(args, result_folder):
    model_family = "bounded_rationality" if args.use_bounded_rationality else "baseline"
    mechanisms = enabled_mechanisms(args)
    metadata_path = os.path.join(result_folder, "metadata.json")
    existing_metadata = _read_json_if_exists(metadata_path)
    existing_runtime = existing_metadata.get("runtime")
    metadata = {
        "model_family": model_family,
        "schema_version": 2,
        "enabled_mechanisms": mechanisms,
        "mechanism_flags": {
            "passenger_entry": bool(args.use_passenger_entry_br),
            "driver_online": bool(args.use_driver_online_br),
        },
        "data_folder": args.data_folder,
        "pricing_alg": args.pricing_alg,
        "mode": args.mode,
        "name": args.name,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "min_price_multiplier": MINP,
        "max_price_multiplier": MAXP,
        "simulation_parameters": {
            "alpha0": 2.0,
            "alpha1": 2.0,
            "alpha2": 0.1,
            "beta1": 0.4,
            "beta2": 0.1,
            "delta": 0.75,
        },
        "bounded_rationality_parameters": {
            "beta_passenger": args.beta_passenger if args.use_passenger_entry_br else None,
            "beta_driver": args.beta_driver if args.use_driver_online_br else None,
            "passenger_base_utility": args.passenger_base_utility if args.use_passenger_entry_br else None,
            "passenger_price_sensitivity": args.passenger_price_sensitivity if args.use_passenger_entry_br else None,
            "passenger_wait_sensitivity": args.passenger_wait_sensitivity if args.use_passenger_entry_br else None,
            "br_random_seed": (
                args.br_random_seed if args.br_random_seed is not None else args.seed
            ) if mechanisms else None,
            "rng_streams": {
                mechanism: "SeedSequence.spawn({})".format(stream_index)
                for mechanism, stream_index, is_enabled in [
                    ("passenger_entry", 0, args.use_passenger_entry_br),
                    ("driver_online", 1, args.use_driver_online_br),
                ]
                if is_enabled
            } if mechanisms else None,
        },
        "process_audit_enabled": bool(args.enable_process_audit),
        "event_order": [
            "match_existing_waiting_requests",
            "apply_configured_passenger_response",
            "execute_matched_services_and_update_vehicles",
            "expire_and_enqueue_requests",
        ],
    }
    spec = ActionSpec(getattr(args, "num_zone", 1), getattr(args, "action_mode", "price_only"),
                      getattr(args, "compensation_min", 0.5), getattr(args, "compensation_max", 2.0))
    metadata["action_space"] = spec.metadata()
    metadata["frequency"] = args.frequency
    if not hasattr(args, "num_zone"):
        metadata["action_space"].update(num_zone=None, dimension=None)
    if existing_runtime is not None:
        metadata["runtime"] = existing_runtime
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

def safe_cuda_synchronize(device):
    device_text = str(device).lower()
    if "cuda" not in device_text:
        return False
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return True
    except Exception as exc:
        print(f"Warning: CUDA synchronize failed: {exc}")
    return False

def build_runtime_info(args, start_time, end_time, start_counter, end_counter, cuda_synchronized):
    elapsed_seconds = max(0.0, end_counter - start_counter)
    return {
        "train_start_time": start_time.replace(microsecond=0).isoformat(),
        "train_end_time": end_time.replace(microsecond=0).isoformat(),
        "train_elapsed_seconds": round(elapsed_seconds, 4),
        "train_elapsed_minutes": round(elapsed_seconds / 60.0, 4),
        "train_elapsed_hours": round(elapsed_seconds / 3600.0, 4),
        "avg_epoch_seconds": round(elapsed_seconds / max(int(args.n_epochs) - int(args.resume), 1), 4),
        "wall_clock_timer": "time.perf_counter",
        "device": args.device,
        "cuda_synchronized": bool(cuda_synchronized),
    }

def update_runtime_metadata(metadata_path, runtime_info):
    metadata = _read_json_if_exists(metadata_path)
    metadata["runtime"] = runtime_info
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

def write_runtime_summary(result_dir, runtime_info, args):
    summary = {
        "data_folder": args.data_folder,
        "pricing_alg": args.pricing_alg,
        "mode": args.mode,
        "name": args.name,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": args.device,
        "use_bounded_rationality": bool(args.use_bounded_rationality),
        "use_passenger_entry_br": bool(args.use_passenger_entry_br),
        "use_driver_online_br": bool(args.use_driver_online_br),
        "enabled_mechanisms": enabled_mechanisms(args),
        "beta_passenger": args.beta_passenger if args.use_passenger_entry_br else None,
        "beta_driver": args.beta_driver if args.use_driver_online_br else None,
        "action_space": ActionSpec(args.num_zone, args.action_mode, args.compensation_min, args.compensation_max).metadata(),
        **runtime_info,
    }
    with open(os.path.join(result_dir, "runtime_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='SpatioTemporal Pricing')
    # Simulation settings
    parser.add_argument('-d', '--data_folder',
                        help='the folder that contains all the input data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-m', '--mode', type=str, default='all',
                        help='running mode, train/all for training or test for testing. (DEFAULT: all)')
    parser.add_argument('-n', '--name', type=str, default='0', help='name_of the run')
    parser.add_argument('-rs', '--seed', type=int, default=47, help='random seed')
    # Training settings
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help='number of epochs (DEFAULT: 10)')
    parser.add_argument('-p', '--pricing_alg', type=str, default='dummy',
                        help='pricing algorithms, can be ddpg_MLP, ddpg_CNN, dummy, or equilibrium')
    parser.add_argument('-c', '--device', type=str, default='cpu',
                        help='cpu or gpu (cuda)')
    parser.add_argument('-f', '--frequency', type=int, default=1,
                        help='pricing freqency')  # update the price every minute
    parser.add_argument('--action_mode', choices=['price_only', 'joint'], default='price_only',
                        help='joint: TD3 controls passenger price and independent order compensation')
    parser.add_argument('--compensation_min', type=float, default=0.5)
    parser.add_argument('--compensation_max', type=float, default=2.0)
    parser.add_argument('-pc', '--policy_constr', action='store_true', default=False,
                        help='constrained pricing change speed')  # constrained policy
    parser.add_argument('-re', '--resume', type=int, default=0,
                        help='resume_prev_training')
    # Hyper parameters for RL algorithms
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size. (DEFAULT: 32)')
    parser.add_argument('-k', '--kernel_size', type=int, default=3,
                        help='kernel size of CNN layer')
    parser.add_argument('-s', '--stride', type=int, default=2,
                        help='stride of CNN layer')
    parser.add_argument('-o', '--od_permutation', action='store_false', default=True,
                        help='enable the permutation of OD features')
    parser.add_argument('-alr', '--actor_lr', type=float, default=0.00001, help='learning rate of actor')
    parser.add_argument('-clr', '--critic_lr', type=float, default=0.001, help='learning rate of critic')
    parser.add_argument('-u', '--update_frequency', type=int, default=25,
                        help='base frequency for policy update') 
    parser.add_argument('-ps', '--pooling', type=int, default=2, help='kernel/stride size of pooling layer for downsampling')
    parser.add_argument('-sa', '--searching', type=str, default='Gaussian', help='searching mechanism')
    parser.add_argument('-pd', '--policy_delay', type = int, default = 0,
                        help='incremental policy delay') # -1: n; 0: n/ln(n); other values: constant
    parser.add_argument('-pe', '--position_encode', type=int, default= 64,\
                        help='whether to use NN for positional encoding')
    parser.add_argument('-ac', '--auto_correlation', action='store_true', default=False,
                        help='whether to use auto-correlated noise for exploration')
    parser.add_argument('-fg', '--forget', action='store_true', default=False,
                        help='whether to use forget in the critic network')
    parser.add_argument('-pm', '--pricing_mode', action='store_true', default=False,
                        help='demand sensitivity to price, either inverse or linear')
    parser.add_argument('--use_bounded_rationality', action='store_true', default=False,
                        help='legacy compatibility switch: enable both passenger-entry and driver-online BR mechanisms')
    parser.add_argument('--use_passenger_entry_br', action='store_true', default=False,
                        help='enable only the passenger-entry bounded-rationality mechanism')
    parser.add_argument('--use_driver_online_br', action='store_true', default=False,
                        help='enable only the driver online/active-supply bounded-rationality mechanism')
    parser.add_argument('--beta_passenger', '--beta_c', dest='beta_passenger', type=float, default=1.0,
                        help='passenger bounded rationality scale; smaller values approach deterministic utility choice')
    parser.add_argument('--beta_driver', '--beta_a', dest='beta_driver', type=float, default=1.0,
                        help='driver bounded rationality scale for online/active decisions')
    parser.add_argument('--passenger_base_utility', '--passenger_value', dest='passenger_base_utility',
                        type=float, default=2.0, help='baseline passenger service utility in the entry logit')
    parser.add_argument('--passenger_price_sensitivity', type=float, default=1.0,
                        help='price coefficient in the passenger entry utility')
    parser.add_argument('--passenger_wait_sensitivity', type=float, default=0.1,
                        help='congestion coefficient in the passenger entry utility')
    parser.add_argument('--br_random_seed', type=int, default=None,
                        help='random seed for bounded rationality choices; defaults to --seed')
    parser.add_argument('--disable_process_audit', dest='enable_process_audit', action='store_false', default=True,
                        help='disable step/zone audit logs and sampled snapshots generated during test runs')
    parser.add_argument('--audit_sample_seed', type=int, default=2026,
                        help='fixed seed for selecting random process-audit snapshot steps')
    parser.add_argument('--audit_random_sample_count', type=int, default=7,
                        help='number of randomly selected process-audit snapshot steps per test log')
    parser.add_argument('--enable_visual_trace', action='store_true', default=False,
                        help='export per-step vehicle/order trace files during test mode')
    parser.add_argument('--trace_dir', type=str, default=None,
                        help='directory for visual trace output; defaults under the selected test result folder')
    parser.add_argument('--trace_test_k', type=int, default=1,
                        help='which test repeat to trace; use 0 to trace all repeats')
    parser.add_argument('--trace_interval', type=int, default=1,
                        help='record one visual trace frame every N simulation steps')
    parser.add_argument('--trace_only', action='store_true', default=False,
                        help='when existing test logs are present, rerun only the traced test without overwriting logs')
    args = normalize_mechanism_flags(parser.parse_args(argv))
    try:
        ActionSpec(1, args.action_mode, args.compensation_min, args.compensation_max)
    except ValueError as exc:
        parser.error(str(exc))
    if args.frequency <= 0 or args.batch_size <= 0 or args.n_epochs <= 0 or args.resume < 0:
        parser.error('frequency, batch_size and n_epochs must be positive; resume must be nonnegative')
    if args.action_mode == 'joint':
        if args.pricing_alg not in {'TD3_MLP', 'TD3_CNN'}:
            parser.error('joint actions require TD3_MLP or TD3_CNN')
        if args.policy_constr:
            parser.error('policy_constr is a legacy price-only option; joint actions use independent box bounds')
        if args.searching not in {'Gaussian', 'Greedy'}:
            parser.error('joint exploration must be Gaussian or Greedy')
        if args.mode in {'train', 'all'} and args.resume >= args.n_epochs:
            parser.error('resume must be smaller than n_epochs')

    return args


if __name__ == '__main__':

    # loading arguments
    args = get_arguments(sys.argv[1:])
    is_train_mode = args.mode in ['all', 'train']

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    permutation_tag = '' if not args.od_permutation else '_perm'
    constraint_tag = '' if not args.policy_constr else '_constr' # unused for this study
    auto_tag = '' if not args.auto_correlation else '_auto'
    forget_tag = '' if not args.forget else '_forget'

    scenario_tag = args.data_folder.replace("/", "_")
    settings_tag = args.pricing_alg + permutation_tag + constraint_tag + auto_tag + forget_tag + "_" +args.searching + "_" +\
                   str(args.policy_delay) +"_"+str(args.position_encode) + "_" + args.name
    settings_tag += bounded_rationality_tag(args)
    if args.action_mode == 'joint':
        settings_tag += '_joint_q{}_{}_f{}'.format(tag_float(args.compensation_min),
            tag_float(args.compensation_max), args.frequency)
    hyperp_tag = str(args.batch_size) + "_" + str(args.actor_lr) + "_" + str(args.critic_lr) + "_" + str(args.kernel_size)

    # prepare the folders for storing the results
    store_run_folder = f"results/{scenario_tag[:-1]}/runs"
    store_model_folder = f"results/{scenario_tag[:-1]}/models/{settings_tag}/{hyperp_tag}" 
    if (args.mode == 'test'):
        store_res_folder = f"results/{scenario_tag[:-1]}/tests/{settings_tag}/{hyperp_tag}"
    else:
        store_res_folder = f"results/{scenario_tag[:-1]}/results/{settings_tag}/{hyperp_tag}"

    if (not os.path.isdir(
            store_run_folder + "/")):
        os.makedirs(
            store_run_folder + "/")

    if (not os.path.isdir(
            store_model_folder + "/")):
        os.makedirs(
            store_model_folder + "/")

    if (not os.path.isdir(
            store_res_folder + "/")):
        os.makedirs(
            store_res_folder + "/")

    args.store_run_folder = store_run_folder
    args.store_model_folder = store_model_folder
    args.store_res_folder = store_res_folder

    if is_train_mode and (os.path.exists(args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv")):
        print("Files are already there, exit!")
        sys.exit()

    if (args.mode == 'test') and (len(os.listdir(args.store_model_folder)) == 0) and (args.pricing_alg not in ['dummy','equilibrium']):
        print("No trained model can be found, exit!")
        sys.exit()

    if (args.mode == 'test') and os.path.exists(args.store_res_folder+"/"+"test_log_1.csv") and (not args.enable_visual_trace):
        print("Files are already there, exit!")
        sys.exit()

    if (args.mode == 'test') and os.path.exists(args.store_res_folder+"/"+"test_log_1.csv") and args.enable_visual_trace and (not args.trace_only):
        print("Test logs are already there. Use --trace_only with --enable_visual_trace to generate trace without overwriting logs.")
        sys.exit()

    write_metadata(args, args.store_res_folder)

    # load data
    if args.verbose:
        print("Loading data")

    # time_tag = datetime.datetime.now().strftime("%m-%d-%H-%M")
    writer = None
    if is_train_mode:
        writer = SummaryWriter(store_run_folder + "/" + settings_tag + "_" + hyperp_tag)

    #td = np.load(args.data_folder + 'td.npy')
    td = np.load(os.path.join(args.data_folder, 'td.npy'))
    tt = np.load(os.path.join(args.data_folder, 'tt.npy'))
    vp = np.load(os.path.join(args.data_folder, 'vp.npy'))
    dd_train = np.load(os.path.join(args.data_folder, 'dd_train.npy'))
    dd_test = np.load(os.path.join(args.data_folder, 'dd_test.npy'))
    permutation = np.load(os.path.join(args.data_folder, 'permutation.npy'))
    permutation2 = np.load(os.path.join(args.data_folder, 'permutation2.npy'))
    args.num_zone = td.shape[0]
    T_train = dd_train.shape[0]
    T_test = dd_test.shape[0]
    write_metadata(args, args.store_res_folder)

    # initialize environment
    if args.verbose:
        print("Initialize environment, vehicle num: " + str(len(vp)))
    if args.action_mode == 'joint':
        controller = None
    elif args.pricing_alg == 'PPO_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling=args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq=args.frequency,
                              veh_num=len(vp), \
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis = 0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'PPO_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='PPO_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'SAC_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='SAC_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling=args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq=args.frequency,
                              veh_num=len(vp), \
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis = 0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'SAC_MLP':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='SAC_MLP',
                              device=args.device,
                              writer=writer, update_freq=args.frequency, veh_num=len(vp),
                              demand_mean=np.mean(dd_train, axis=0), demand_std=np.std(dd_train, axis=0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching=args.searching,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    elif args.pricing_alg == 'TD3_CNN':
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, option='TD3_CNN',
                              permutation=permutation, permutation2=permutation2, kernel_size=args.kernel_size,
                              stride=args.stride, pooling = args.pooling, device=args.device, writer=writer,
                              od_permutation=args.od_permutation, update_freq = args.frequency, veh_num = len(vp),\
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )
    else:
        controller = Platform(td, tt, args.num_zone, actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=args.device,
                              writer=writer, update_freq = args.frequency, veh_num= len(vp),
                              demand_mean = np.mean(dd_train, axis = 0), demand_std = np.std(dd_train, axis = 0),
                              wage_mean=np.mean(vp, axis=0), wage_std=np.std(vp, axis=0),
                              searching = args.searching, policy_delay = args.policy_delay,
                              position_encode=args.position_encode,
                              auto_correlation=args.auto_correlation,
                              forget=args.forget,
                              T_train = T_train
                              )

    env = Environment(td, tt, vp, args.num_zone, frequency=args.frequency, br_config=args)

    if args.action_mode == 'joint':
        # This separate runner leaves legacy replay and decision scheduling intact.
        from joint_training import train_joint, evaluate_joint
        controller = Platform(td, tt, args.num_zone, option=args.pricing_alg,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr, device=args.device, writer=writer,
            update_freq=args.frequency, veh_num=len(vp), demand_mean=np.mean(dd_train, axis=0),
            demand_std=np.std(dd_train, axis=0), wage_mean=np.mean(vp), wage_std=np.std(vp),
            permutation=permutation, permutation2=permutation2, od_permutation=args.od_permutation,
            kernel_size=args.kernel_size, stride=args.stride, pooling=args.pooling,
            searching=args.searching, policy_delay=args.policy_delay, position_encode=args.position_encode,
            auto_correlation=args.auto_correlation, forget=args.forget, T_train=T_train,
            action_mode='joint', compensation_min=args.compensation_min, compensation_max=args.compensation_max)
        if is_train_mode:
            start_sync = safe_cuda_synchronize(args.device)
            start_time, start_counter = datetime.now(), time.perf_counter()
            train_joint(env, controller, dd_train, writer, args)
            end_sync = safe_cuda_synchronize(args.device)
            runtime = build_runtime_info(args, start_time, datetime.now(), start_counter,
                                        time.perf_counter(), start_sync and end_sync)
            update_runtime_metadata(os.path.join(args.store_res_folder, 'metadata.json'), runtime)
            write_runtime_summary(args.store_res_folder, runtime, args)
            writer.close()
        else:
            evaluate_joint(env, controller, dd_train, dd_test, args)
        sys.exit(0)

    # input: travel distance, travel time matrices, initial vehicle distribution
    # output: time step, profit, reposition cost
    if is_train_mode:
        if args.resume > 0:
            if args.verbose:
                print("Load model")
            print("Resume training from epoch " + str(args.resume))
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.resume)
            controller.buffer.load(args.store_model_folder + "/", args.resume)
            if args.device == 'cuda':
                controller.pricer.cuda()

            # update the searching variance
            for i in range(args.resume):
                controller.decay_searching_variance()

        baseline = (1 - env.delta) * (env.alpha0 * np.sum(dd_train, axis=(1, 2)) + env.alpha1 * np.sum(
            dd_train * td[None, :, :], axis=(1, 2)) + \
                   env.beta1 * np.sum(dd_train * tt[None, :, :], axis=(1, 2)))
        start_sync = safe_cuda_synchronize(args.device)
        train_start_time = datetime.now()
        train_start_counter = time.perf_counter()
        args.train_timer_start = train_start_counter
        train(env, controller, dd_train, T_train, baseline, writer, args)
        end_sync = safe_cuda_synchronize(args.device)
        train_end_counter = time.perf_counter()
        train_end_time = datetime.now()
        runtime_info = build_runtime_info(
            args,
            train_start_time,
            train_end_time,
            train_start_counter,
            train_end_counter,
            start_sync and end_sync,
        )
        update_runtime_metadata(os.path.join(args.store_res_folder, "metadata.json"), runtime_info)
        write_runtime_summary(args.store_res_folder, runtime_info, args)
        # test(env, controller, dd_train, dd_test, T_test, args, 0)
    else:
        # load the model
        if args.verbose:
            print("Load model")
        if args.pricing_alg.startswith('TD3'):
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.n_epochs)
            if args.device == 'cuda':
                controller.pricer.cuda()
        if args.pricing_alg.startswith('PPO'):
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.n_epochs)
            controller.pricer.set_action_std(controller.min_action_std)
            if args.device == 'cuda':
                controller.pricer.cuda()
        if args.pricing_alg.startswith('SAC'):
            controller.pricer.load_weights(
                args.store_model_folder + "/", args.n_epochs)
            controller.pricer.set_action_std(controller.min_action_std)
            if args.device == 'cuda':
                controller.pricer.cuda()
        # run exp
        test_repeats = range(1, 6)
        if args.enable_visual_trace and args.trace_only and args.trace_test_k > 0:
            test_repeats = [args.trace_test_k]
        for k in test_repeats:
            test(env, controller, dd_train, dd_test, T_test, args, k)
    # collect result
    if is_train_mode:
        writer.close()
