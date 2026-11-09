import numpy as np
import pandas as pd
import torch
import os
import time

from visual_trace import TraceRecorder
from process_audit import ProcessAuditRecorder

def price_sensitivity(pricing_multipliers, mode = False):
    # A monotonic decreasing function that describe the price elasticity
    # To make the problem challenging (and more fun), we choose the nonlinear function such that the increase of profit
    # can only come from the improvement of service efficiency (serving more passengers) with less empty mileage
    if mode:
        return max(2.0 - pricing_multipliers, 0)
    else:
        return 1.0 / pricing_multipliers

BR_LOG_COLUMNS = [
    "potential_passenger_count",
    "actual_passenger_count",
    "passenger_enter_rate",
    "passenger_balk_count",
    "potential_driver_count",
    "active_driver_count",
    "driver_online_rate",
    "matched_order_count",
    "served_order_count",
]

def _sample_fractional_counts(expected_counts):
    expected_counts = np.maximum(np.asarray(expected_counts, dtype=float), 0.0)
    base_counts = np.floor(expected_counts).astype(int)
    residual = np.clip(expected_counts - base_counts, 0.0, 1.0)
    return base_counts + np.random.binomial(1, residual)

def _sample_poisson_counts(demand_rates):
    demand_rates = np.asarray(demand_rates, dtype=float)
    sampled = np.zeros(demand_rates.shape, dtype=int)
    ind = demand_rates > 0
    sampled[ind] = np.random.poisson(demand_rates[ind])
    return sampled

def _price_sensitive_counts(base_counts, pricing_multipliers, args):
    expected = base_counts * price_sensitivity(pricing_multipliers[None, :, None], args.pricing_mode)
    return _sample_fractional_counts(expected)

def _passenger_entry_enabled(args):
    if hasattr(args, "use_passenger_entry_br"):
        return bool(args.use_passenger_entry_br)
    return bool(getattr(args, "use_bounded_rationality", False))

def _prepare_demand_batch(base_counts, pricing_multipliers, args):
    if _passenger_entry_enabled(args):
        return _sample_fractional_counts(base_counts)
    return _price_sensitive_counts(base_counts, pricing_multipliers, args)

def _activate_demand(env, demand_batch, batch_index, pricing_multipliers):
    return env.apply_passenger_bounded_rationality(demand_batch[batch_index], pricing_multipliers)

def _zero_br_stats():
    return {key: 0.0 for key in BR_LOG_COLUMNS}

def _accumulate_br_stats(total, stats):
    for key in BR_LOG_COLUMNS:
        total[key] += float(stats.get(key, 0.0))

def _finalize_br_stats(total):
    res = dict(total)
    res["passenger_enter_rate"] = total["actual_passenger_count"] / (total["potential_passenger_count"] + 1e-8)
    res["driver_online_rate"] = total["active_driver_count"] / (total["potential_driver_count"] + 1e-8)
    return res

def train(env, controller, dd_train, T_train, baseline, writer, args):
    thrng = torch.Generator("cpu")
    # print(np.sum(baseline))
    if args.resume == 0:
        controller.pricer.initialize(thrng)
    if args.device == 'cuda':
        controller.pricer.cuda()
    if args.verbose:
        print("Start training")

    res = []  # pd.DataFrame(columns = 'epoch', 't', 'policy', 'profit', 'welfare')

    # [新增] 确保 temp_p 在任何情况下都有初始值
    # dd_train的第二维是区域数量 (num_zone)
    num_zone = dd_train.shape[1]
    temp_p = np.ones(num_zone) # 初始化定价乘数为 1.0 (中性定价)

    # one day to warm up to obtain the initial active vehicle distribution
    for t in range(max(T_train-1440, 0), T_train):
        # print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        ongoing_veh = env.get_ongoing_veh()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                # no need to train
                #...
                # FIX: Assign temp_p for dummy case
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                #...
                # FIX: Assign temp_p for equilibrium case
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh ,
                                                          [dd_train[(t + i) % T_train, :, :] for i in range(10)], args.data_folder)
            else:
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                             ongoing_veh, env.avg_profit, env.zone_profit,
                                             env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                             t, 'train',
                                             od_permutation=args.od_permutation,
                                             policy_constr=args.policy_constr,
                                             last_pricing=env.pricing_multipliers[0, :])
            dd_train_ = _sample_poisson_counts(dd_train[t:(t + args.frequency), :, :])
            temp_demands = _prepare_demand_batch(dd_train_, temp_p, args)
        temp_demand = _activate_demand(env, temp_demands, t % args.frequency, temp_p)
        env.step(temp_demand, temp_schedule, temp_p)

    for e in range(args.resume, args.n_epochs):
        epoch_start_counter = time.perf_counter()
        total_time_step = 0
        total_profit = 0
        total_reposition = 0
        total_expense = 0
        total_served_pass = 0
        total_left_pass = 0

        total_occu_mile = 0
        total_occu_minute = 0
        total_empty_mile = 0
        total_empty_minute = 0
        total_active_veh = 0

        temp_reward = 0
        temp_demands = np.zeros((args.frequency, args.num_zone))
        total_br_stats = _zero_br_stats()

        if args.verbose:
            print("Epoch " + str(e))
        for t in range(T_train):
            with torch.no_grad():
                temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                                     env.veh_count)
                if t % args.frequency == 0:
                    if args.pricing_alg == 'dummy':
                        # no need to train
                        ...
                    elif args.pricing_alg == 'equilibrium':
                        ...
                    else:
                        ongoing_veh = env.get_ongoing_veh()
                        controller.add_memory(env.pass_count, env.veh_count, ongoing_veh, env.avg_profit, env.zone_profit,\
                                              env.pricing_multipliers[0:max(5 // args.frequency, 1), :], temp_reward, t,
                                              args.od_permutation)
                        temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'train',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])

                    dd_train_ = _sample_poisson_counts(dd_train[t:(t + args.frequency), :, :])
                    temp_demands = _prepare_demand_batch(dd_train_, temp_p, args)
                    temp_reward = 0  # refresh the temp_reward

                temp_demand = _activate_demand(env, temp_demands, t % args.frequency, temp_p)
                temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
                temp_reward += temp_profit - temp_expense - baseline[t]
                step_br_stats = env.get_last_step_stats()
            # train the model
            # if t == 0 and e == 1:
            #     controller.train_pricing_value(1000, args.batch_size)
            if args.pricing_alg.startswith('TD3') and (e * T_train + t + 1) % (args.frequency) == 0: 
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //args.frequency, \
                                                T_train // args.frequency, args.batch_size, args.update_frequency)
            elif args.pricing_alg.startswith('PPO') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(50, \
                                                args.update_frequency, args.batch_size)
            elif args.pricing_alg.startswith('SAC') and (e * T_train + t + 1) % (args.frequency * args.update_frequency) == 0:
                controller.train_pricing_policy(((e - 1) * T_train + t + 1) //(args.frequency * args.update_frequency), \
                                                args.update_frequency, args.batch_size)

            total_time_step += 1
            total_profit += temp_profit
            total_reposition += temp_cost
            total_expense += temp_expense
            total_served_pass += temp_served_pass
            total_left_pass += temp_left_pass
            total_occu_mile += occu_mile
            total_occu_minute += occu_minute
            total_empty_mile += empty_mile
            total_empty_minute += empty_minute
            total_active_veh += env.active_veh
            _accumulate_br_stats(total_br_stats, step_br_stats)

            if t % 120 == 0:
                writer.flush()
                if args.verbose:
                    # print(np.diag(env.veh_count))
                    print("Time step: " + str(t) + ", income: " + str(temp_profit) + \
                          ", expense: " + str(temp_expense) + \
                          ", active veh: " + str(env.active_veh) + \
                          ", served_pass: " + str(temp_served_pass) + \
                          ", left_pass: " + str(temp_left_pass) + \
                          ", new demand: " + str(np.sum(temp_demand)))

        controller.decay_searching_variance()
        print("Total time step: " + str(total_time_step) + ", reward: " + str(
            total_profit - total_expense) + ", profit: " + str(total_profit) + \
              ", expense: " + str(total_expense) + ", served_pass: " + str(
            total_served_pass) + ", left_pass: " + str(total_left_pass))
        epoch_br_stats = _finalize_br_stats(total_br_stats)
        epoch_elapsed_seconds = time.perf_counter() - epoch_start_counter
        train_start_counter = getattr(args, "train_timer_start", None)
        elapsed_seconds = (
            time.perf_counter() - train_start_counter
            if train_start_counter is not None
            else epoch_elapsed_seconds
        )
        completed_epochs = max(e - args.resume + 1, 1)
        avg_epoch_seconds = elapsed_seconds / completed_epochs
        res.append([e, total_profit, total_expense, total_served_pass, total_left_pass,\
                    total_occu_mile, total_occu_minute, total_empty_mile, total_empty_minute,
                    total_profit - total_expense, total_active_veh / (total_time_step + 1e-8),
                    elapsed_seconds, epoch_elapsed_seconds, avg_epoch_seconds] +
                   [epoch_br_stats[key] for key in BR_LOG_COLUMNS])
        if env.active_veh == 0: # run into an absorbing stage in which no drive wants to work, need to reset the env
            env.reset()
            controller.buffer.clear()
        writer.add_scalar("Reward/train", total_profit - total_expense, e)
        writer.add_scalar("Served_passenger/train", total_served_pass, e)
        writer.add_scalar("Left_passenger/train", total_left_pass, e)

    res = pd.DataFrame(
        res,
        columns=['epoch', 'income', 'expense', 'served', 'left', 'occu_mile', 'occu_minute',
                 'empty_mile', 'empty_minute', 'reward', 'act_veh',
                 'elapsed_seconds', 'epoch_elapsed_seconds', 'avg_epoch_seconds'] + BR_LOG_COLUMNS
    )
    res.to_csv(
        args.store_res_folder+"/"+"train_log_"+str(args.n_epochs)+".csv",
        index=None)

    # save the model in the end
    controller.pricer.save_model(
        args.store_model_folder + "/", args.n_epochs)
    controller.buffer.save(args.store_model_folder + "/", args.n_epochs)

    # store the model
    # print("Store model")
    # if (not os.path.isdir(
    #         'models/' + args.name+'/'+ args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #             args.batch_size) + "_" + str(
    #             args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")):
    #     os.makedirs('models/' +  args.name+'/'+args.data_folder + args.pricing_alg + permutation_tag + "/" + str(
    #         args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")
    # controller.pricer.save_model(
    #     'models/' +args.name+'/' + args.data_folder + args.pricing_alg + permutation_tag + "/" + str(args.batch_size) + "_" + str(
    #         args.actor_lr) + "_" + str(args.critic_lr) + str(args.epoch) + "/")

def test(env, controller, dd_train, dd_test, T_test, args, k):
    if args.verbose:
        print("Start testing")
    res = []
    trace_recorder = None
    audit_recorder = None
    test_log_path = args.store_res_folder + "/" + f"test_log_{k}.csv"
    write_test_outputs = (not getattr(args, "trace_only", False)) or (not os.path.exists(test_log_path))
    if getattr(args, "enable_process_audit", True) and write_test_outputs:
        audit_recorder = ProcessAuditRecorder(
            args.store_res_folder,
            k,
            T_test,
            random_sample_count=getattr(args, "audit_random_sample_count", 7),
            sample_seed=getattr(args, "audit_sample_seed", 2026),
        )
    trace_test_k = int(getattr(args, "trace_test_k", 1))
    if getattr(args, "enable_visual_trace", False) and (trace_test_k <= 0 or k == trace_test_k):
        trace_dir = getattr(args, "trace_dir", None)
        if not trace_dir:
            trace_dir = os.path.join(args.store_res_folder, "visual_trace", f"test_log_{k}")
        metadata = {
            "data_folder": args.data_folder,
            "pricing_alg": args.pricing_alg,
            "name": args.name,
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "test_log": f"test_log_{k}.csv",
            "store_res_folder": args.store_res_folder,
            "num_zone": args.num_zone,
            "frequency": args.frequency,
            "use_bounded_rationality": bool(args.use_bounded_rationality),
            "use_passenger_entry_br": bool(args.use_passenger_entry_br),
            "use_driver_online_br": bool(args.use_driver_online_br),
            "enabled_mechanisms": [
                mechanism for mechanism, enabled in [
                    ("passenger_entry", args.use_passenger_entry_br),
                    ("driver_online", args.use_driver_online_br),
                ] if enabled
            ],
            "model_family": "bounded_rationality" if args.use_bounded_rationality else "baseline",
            "schema_version": 2,
        }
        trace_recorder = TraceRecorder(
            trace_dir,
            metadata=metadata,
            trace_interval=getattr(args, "trace_interval", 1),
        )
    controller.pricer.eval()
    total_time_step = 0
    total_profit = 0
    total_reposition = 0
    total_expense = 0
    total_served_pass = 0
    total_left_pass = 0
    temp_demands = np.zeros((args.frequency, args.num_zone))

    for t in range(max(T_test-1440, 0), T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        # temp_p = controller.dummy_price()
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                # no need to train
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)), env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                 ongoing_veh, env.avg_profit, env.zone_profit,
                                                 env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                 t, 'train',
                                                 od_permutation=args.od_permutation,
                                                 policy_constr=args.policy_constr,
                                                 last_pricing=env.pricing_multipliers[0, :])
        dd_train_ = np.random.poisson(dd_train[t, :, :]).astype(int)
        temp_demand = dd_train_
        if _passenger_entry_enabled(args):
            temp_demand = env.apply_passenger_bounded_rationality(temp_demand, temp_p)
        # temp_demand += np.random.binomial(1, (dd_train_ / temp_p[:, None] - temp_demand))
        env.step(temp_demand, temp_schedule, temp_p)

    for t in range(T_test):
        print(t)
        temp_cost, temp_schedule = controller.batch_matching(np.sum(env.pass_count, axis=(0, 2)),
                                                             env.veh_count)
        if t % args.frequency == 0:
            if args.pricing_alg == 'dummy':
                temp_p = controller.dummy_price()
            elif args.pricing_alg == 'equilibrium':
                temp_p = controller.equilibrium_price(np.sum(env.pass_count, axis=(0)),env.active_veh ,
                                                          [dd_train[(t + i) % T_test, :, :] for i in range(10)], args.data_folder)
            else:
                ongoing_veh = env.get_ongoing_veh()
                temp_p = controller.update_price(env.pass_count, env.veh_count, \
                                                         ongoing_veh, env.avg_profit, env.zone_profit,
                                                         env.pricing_multipliers[0:max(5 // args.frequency, 1), :],
                                                         t, 'test',
                                                         od_permutation=args.od_permutation,
                                                         policy_constr=args.policy_constr,
                                                         last_pricing=env.pricing_multipliers[0, :])
            temp_demands = _prepare_demand_batch(dd_test[t:(t + args.frequency), :, :], temp_p, args)

        temp_demand = _activate_demand(env, temp_demands, t % args.frequency, temp_p)
        temp_profit, temp_expense, temp_served_pass, temp_left_pass, occu_mile, occu_minute, empty_mile, empty_minute = env.step(temp_demand, temp_schedule, temp_p)
        step_br_stats = env.get_last_step_stats()
        step_metrics = {
            "profit": temp_profit,
            "expense": temp_expense,
            "act_veh": env.active_veh,
            "served": temp_served_pass,
            "left": temp_left_pass,
            "occu_mile": occu_mile,
            "occu_minute": occu_minute,
            "empty_mile": empty_mile,
            "empty_minute": empty_minute,
            "reward": temp_profit - temp_expense,
            "price_mean": float(np.mean(temp_p)),
            "price_min": float(np.min(temp_p)),
            "price_max": float(np.max(temp_p)),
        }
        total_time_step += 1
        total_profit += temp_profit
        total_reposition += temp_cost
        total_expense += temp_expense
        total_served_pass += temp_served_pass
        total_left_pass += temp_left_pass

        # writer.add_scalar("Reward/test", temp_profit + temp_welfare, t)
        # writer.add_scalar("Served_passenger/test", temp_served_pass, t)
        # writer.add_scalar("Left_passenger/test", temp_left_pass, t)

        if trace_recorder is not None:
            trace_recorder.record_step(t, temp_p, step_metrics, env)
        if audit_recorder is not None:
            audit_recorder.record_step(t, temp_p, step_metrics, env)

        res.append([t, temp_p, step_metrics["profit"], step_metrics["expense"], step_metrics["act_veh"],
                    step_metrics["served"], step_metrics["left"], step_metrics["occu_mile"],
                    step_metrics["occu_minute"], step_metrics["empty_mile"], step_metrics["empty_minute"],
                    step_metrics["reward"], step_metrics["price_mean"], step_metrics["price_min"],
                    step_metrics["price_max"]] +
                   [step_br_stats[key] for key in BR_LOG_COLUMNS])
        if t % 120 == 0:
            # writer.flush()
            if args.verbose:
                # print(np.diag(env.veh_count))
                print("Time step: " + str(t) + ", profit: " + str(temp_profit) + ", expense: " + str(
                    temp_expense) + \
                      ", active veh: " + str(env.active_veh) + \
                      ", served_pass: " + str(temp_served_pass) + \
                      ", left_pass: " + str(temp_left_pass) + \
                      ", new demand: " + str(np.sum(temp_demand)))
    print("Total time step: " + str(total_time_step) + ", reward: " + str(
        total_profit - total_expense) + ", profit: " + str(total_profit) + \
          ", expense: " + str(total_expense) + ", served_pass: " + str(
        total_served_pass) + ", left_pass: " + str(
        total_left_pass))
    res = pd.DataFrame(
        res,
        columns=['t', 'policy', 'profit', 'expense', 'act_veh', 'served', 'left',
                 'occu_mile', 'occu_minute', 'empty_mile', 'empty_minute', 'reward',
                 'price_mean', 'price_min', 'price_max'] + BR_LOG_COLUMNS
    )
    if write_test_outputs:
        res.to_csv(test_log_path, index=None)
    if trace_recorder is not None:
        trace_recorder.close()
    if audit_recorder is not None:
        audit_recorder.close()
    env.reset()
