import copy
from collections import deque
import numpy as np
from economic_action import validate_control
from sampling import sample_fractional_counts

class Environment:
    def __init__(self, travel_distance, travel_time, veh_profile, num_zone, max_waiting=5, frequency=1, br_config=None):
        assert num_zone == travel_distance.shape[0] and num_zone == travel_time.shape[0]
        self.travel_distance = travel_distance
        self.travel_time = travel_time
        self.veh_profiles = veh_profile
        self.num_zone = num_zone
        self.max_waiting = max_waiting
        self.frequency = frequency
        self.pricing_multipliers = np.ones((max_waiting, num_zone))
        self.compensation_multipliers = self.pricing_multipliers.copy()
        self.action_mode = getattr(br_config, "action_mode", "price_only")

        # parameters, alpha2 and beta2 selected based on the min wage of ridehailing drivers
        # https://www.engadget.com/nyc-gig-drivers-pay-increase-012304976.html
        # https://www.ridester.com/uber-rates-cost/
        self.alpha0 = 2 # base fare for passengers
        self.alpha1 = 2 # payments $2/ miles from passengers
        self.alpha2 = 0.1 # 0.1 # $welfare cost for empty mile, set 0 when targeting only the platform profit

        self.beta1 = 0.4 # payment $0.4/ minutes from passengers
        self.beta2 = 0.1 # 0.1 # welfare cost for empty minute

        self.delta = 0.75 # 3/4 passenger payments goes to drivers
        self.configure_bounded_rationality(br_config)

        # data for sim
        self.pass_count = np.zeros((max_waiting, num_zone, num_zone), dtype=int)
        self.veh_count = np.zeros(num_zone, dtype=int)
        self.veh_queue = [deque() for i in range(num_zone)]  # queue for waiting vehicles
        self.veh_list = []  # list of all vehicles
        self.avg_profits = [np.mean(
            veh_profile)] * 30  # active or not depends on the average profit, default_profit is average veh_profit
        self.zone_profits = np.ones((30, num_zone))
        self.avg_profit = np.mean(veh_profile)
        self.zone_profit = np.ones(num_zone)
        self.active_veh = 0
        # initialization
        self.initialize_veh()

        if self.env_rng_mode == "legacy":
            self.rand_int = list(np.random.randint(0,self.num_zone,size=100000))
            self.rand_double = list(np.random.random(size=100000))
        else:
            self.rand_int, self.rand_double = [], []

    # market-rule fields parsed by configure_market_rules; a snapshot carries them in its __dict__
    MARKET_RULE_FIELDS = (
        "use_passenger_entry_br", "use_driver_online_br", "use_bounded_rationality", "driver_decision_interval",
        "driver_switch_cost", "passenger_calibration", "online_zone_choice", "driver_signal", "idle_cost_bearer",
        "target_demand_elasticity", "passenger_anchor_utility", "passenger_anchor_price", "beta_passenger",
        "beta_driver", "passenger_base_utility", "passenger_price_sensitivity", "passenger_wait_sensitivity",
    )

    def configure_market_rules(self, config=None):
        """Parse the market rules (switches, calibration, driver inertia, utility parameters) from ``config``.

        Touches no random stream and no simulation state, so it can also re-apply a market to a restored snapshot.
        """
        has_individual_flags = config is not None and (
            hasattr(config, "use_passenger_entry_br")
            or hasattr(config, "use_driver_online_br")
        )
        if has_individual_flags:
            self.use_passenger_entry_br = bool(
                getattr(config, "use_passenger_entry_br", False)
            )
            self.use_driver_online_br = bool(
                getattr(config, "use_driver_online_br", False)
            )
        else:
            legacy_all_on = bool(getattr(config, "use_bounded_rationality", False))
            self.use_passenger_entry_br = legacy_all_on
            self.use_driver_online_br = legacy_all_on
        self.use_bounded_rationality = bool(
            self.use_passenger_entry_br or self.use_driver_online_br
        )
        # driver decision inertia: driver k re-decides only when (step + k) % interval == 0
        self.driver_decision_interval = int(getattr(config, "driver_decision_interval", 1))
        self.driver_switch_cost = float(getattr(config, "driver_switch_cost", 0.0))
        if self.driver_decision_interval < 1 or self.driver_switch_cost < 0:
            raise ValueError("driver_decision_interval must be >= 1 and driver_switch_cost >= 0")
        # passenger entry calibration: "none" (legacy logit) or "anchor"
        self.passenger_calibration = getattr(config, "passenger_calibration", "none")
        if self.passenger_calibration not in ("none", "anchor"):
            raise ValueError("passenger_calibration must be none or anchor")
        self.online_zone_choice = getattr(config, "online_zone_choice", "random")
        if self.online_zone_choice not in ("random", "pay_weighted"):
            raise ValueError("online_zone_choice must be random or pay_weighted")
        # online/offline income signal: "gross" (legacy) or "net" = minus idle cruising and reposition costs
        self.driver_signal = getattr(config, "driver_signal", "gross")
        if self.driver_signal not in ("gross", "net"):
            raise ValueError("driver_signal must be gross or net")
        self.idle_cost_bearer = getattr(config, "idle_cost_bearer", "platform")
        if self.idle_cost_bearer not in ("platform", "driver"):
            raise ValueError("idle_cost_bearer must be platform or driver")
        self.target_demand_elasticity = float(getattr(config, "target_demand_elasticity", -1.0))
        self.passenger_anchor_utility = float(getattr(config, "passenger_anchor_utility", 1.0))
        self.passenger_anchor_price = float(getattr(config, "passenger_anchor_price", 1.0))
        self.beta_passenger = float(getattr(config, "beta_passenger", 1.0))
        self.beta_driver = float(getattr(config, "beta_driver", 1.0))
        self.passenger_base_utility = float(getattr(config, "passenger_base_utility", 2.0))
        self.passenger_price_sensitivity = float(getattr(config, "passenger_price_sensitivity", 1.0))
        self.passenger_wait_sensitivity = float(getattr(config, "passenger_wait_sensitivity", 0.1))
        if self.passenger_calibration == "anchor":
            if self.beta_passenger <= 1e-8:
                raise ValueError("anchor calibration needs beta_passenger > 0")
            self._configure_anchor_calibration()
        else:  # a restored snapshot may carry the previous market's calibrated values
            for stale in ("calibrated_price_sensitivity", "calibrated_base_utility", "calibration_anchor_probability"):
                self.__dict__.pop(stale, None)

    def market_rule_values(self):
        return {name: getattr(self, name) for name in self.MARKET_RULE_FIELDS}

    def apply_market(self, config):
        """Overwrite the market rules of this (e.g. snapshot-restored) environment; simulation state is kept."""
        self.configure_market_rules(config)

    def configure_bounded_rationality(self, config=None):
        self.configure_market_rules(config)
        seed = getattr(config, "br_random_seed", None)
        if seed is None:
            seed = getattr(config, "seed", None)
        self.br_random_seed = seed
        seed_sequence = np.random.SeedSequence(seed)
        # children 0 and 1 are identical to spawn(2); child 2 only exists for env_rng="isolated"
        passenger_seed, driver_seed, env_seed = seed_sequence.spawn(3)
        self.passenger_rng = np.random.default_rng(passenger_seed)
        self.driver_rng = np.random.default_rng(driver_seed)
        self.env_rng_mode = getattr(config, "env_rng", "legacy")
        if self.env_rng_mode not in ("legacy", "isolated"):
            raise ValueError("env_rng must be legacy or isolated")
        self.env_rng = np.random.default_rng(env_seed) if self.env_rng_mode == "isolated" else None
        self.step_index = 0
        self._last_reposition_expense = 0.0

        self._pending_passenger_stats = None
        self._pending_passenger_audit = None
        self.last_step_stats = self._empty_step_stats()
        self.last_step_audit = {}
        self._next_trace_order_id = 0
        self.last_step_trace = self._empty_step_trace()

    def _empty_step_stats(self):
        return {
            "potential_passenger_count": 0.0,
            "actual_passenger_count": 0.0,
            "passenger_enter_rate": 0.0,
            "passenger_balk_count": 0.0,
            "potential_driver_count": float(len(self.veh_profiles)),
            "active_driver_count": 0.0,
            "driver_online_rate": 0.0,
            "matched_order_count": 0.0,
            "served_order_count": 0.0,
            "driver_decision_count": 0.0,
            "driver_expected_online_decision_count": 0.0,
            "driver_actual_online_decision_count": 0.0,
            "driver_online_decision_variance": 0.0,
        }

    def _empty_step_trace(self):
        return {
            "order_events": [],
            "vehicle_events": [],
        }

    def _sigmoid_probability(self, utility, beta):
        utility = np.asarray(utility, dtype=float)
        if beta <= 1e-8:
            return (utility >= 0).astype(float)
        z = np.clip(utility / beta, -60, 60)
        return 1.0 / (1.0 + np.exp(-z))

    def _configure_anchor_calibration(self):
        """Entry = D * sigmoid((a - b p - gamma rho^2)/beta) / sigmoid(c/beta) with a - b p0 = c.

        b is chosen so the entry elasticity at (p0, rho=0) equals the target:
        b = -e * beta / (p0 * (1 - sigmoid(c / beta))); for beta = 1 this is -e / (p0 (1 - sigmoid(c))).
        """
        c, p0, beta = self.passenger_anchor_utility, self.passenger_anchor_price, self.beta_passenger
        if p0 <= 0 or self.target_demand_elasticity >= 0:
            raise ValueError("anchor calibration needs p0 > 0 and a negative target elasticity")
        anchor_prob = float(self._sigmoid_probability(c, beta))
        self.calibrated_price_sensitivity = -self.target_demand_elasticity * beta / (p0 * (1.0 - anchor_prob))
        self.calibrated_base_utility = c + self.calibrated_price_sensitivity * p0
        self.calibration_anchor_probability = anchor_prob

    def _passenger_decision_context(self, price_multiplier):
        waiting_by_origin = np.sum(self.pass_count, axis=(0, 2))
        idle_supply = self.veh_count.copy()
        congestion = waiting_by_origin / (idle_supply + 1.0)
        if self.passenger_calibration == "anchor":
            utility = (
                self.calibrated_base_utility
                - self.calibrated_price_sensitivity * np.asarray(price_multiplier)
                - self.passenger_wait_sensitivity * congestion ** 2
            )
            enter_probability = self._sigmoid_probability(utility, self.beta_passenger) / self.calibration_anchor_probability
        else:
            utility = (
                self.passenger_base_utility
                - self.passenger_price_sensitivity * np.asarray(price_multiplier)
                - self.passenger_wait_sensitivity * congestion ** 2
            )
            enter_probability = self._sigmoid_probability(utility, self.beta_passenger)
        return {
            "waiting_by_origin": waiting_by_origin.astype(float),
            "idle_supply": idle_supply.astype(float),
            "congestion_proxy": congestion.astype(float),
            "passenger_utility": utility.astype(float),
            "passenger_enter_probability": enter_probability,
        }

    def passenger_enter_probability(self, price_multiplier):
        return self._passenger_decision_context(price_multiplier)["passenger_enter_probability"]

    def apply_passenger_bounded_rationality(self, potential_demand, price_multiplier):
        potential_demand = np.asarray(potential_demand, dtype=int)
        potential_demand = np.maximum(potential_demand, 0)
        potential_count = float(np.sum(potential_demand))
        context = self._passenger_decision_context(price_multiplier)
        if self.use_passenger_entry_br:
            enter_prob = context["passenger_enter_probability"]
            if self.passenger_calibration == "anchor":
                # the calibrated entry ratio may exceed 1 at low prices, as in the legacy price rule
                actual_demand = sample_fractional_counts(potential_demand * enter_prob[:, None], self.passenger_rng)
            else:
                actual_demand = self.passenger_rng.binomial(potential_demand, enter_prob[:, None])
        else:
            enter_prob = np.ones(self.num_zone, dtype=float)
            actual_demand = potential_demand

        actual_count = float(np.sum(actual_demand))
        self._pending_passenger_stats = {
            "potential_passenger_count": potential_count,
            "actual_passenger_count": actual_count,
            "passenger_enter_rate": actual_count / (potential_count + 1e-8),
            "passenger_balk_count": potential_count - actual_count,
        }
        self._pending_passenger_audit = {
            **context,
            "passenger_enter_probability": enter_prob.astype(float),
            "potential_demand_matrix": potential_demand.copy(),
            "actual_demand_matrix": actual_demand.astype(int).copy(),
        }
        return actual_demand.astype(int)

    def driver_online_probability(self, driver_profile, switch_offset=0.0):
        utility = self.avg_profit - driver_profile
        if switch_offset:
            utility = utility + switch_offset
        return float(self._sigmoid_probability(utility, self.beta_driver))

    def _choose_online_zone(self, uniform):
        """Zone of a driver who goes online: uniform (legacy) or weighted by the 30-minute zone pay signal."""
        if self.online_zone_choice == "pay_weighted":
            if uniform is None:
                if len(self.rand_double) == 0:
                    self.rand_double = list(np.random.random(size=100000))
                uniform = self.rand_double.pop()
            cumulative = np.cumsum(self.zone_profit + 1e-4)
            return int(min(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"), self.num_zone - 1))
        if uniform is not None:
            return min(int(uniform * self.num_zone), self.num_zone - 1)
        return self.rand_int.pop()

    def _driver_decides(self, vehicle):
        return self.driver_decision_interval == 1 or (self.step_index + vehicle.id) % self.driver_decision_interval == 0

    def _hard_online(self, profile, switch_offset):
        return profile <= (self.avg_profit + switch_offset if switch_offset else self.avg_profit)

    def get_last_step_stats(self):
        return dict(self.last_step_stats)

    def get_last_step_trace(self):
        return {
            "order_events": list(self.last_step_trace.get("order_events", [])),
            "vehicle_events": list(self.last_step_trace.get("vehicle_events", [])),
        }

    def get_last_step_audit(self):
        audit = {}
        for key, value in self.last_step_audit.items():
            audit[key] = value.copy() if isinstance(value, np.ndarray) else value
        return audit

    def _vehicle_vectors(self):
        return (
            np.asarray([v.state for v in self.veh_list], dtype=int),
            np.asarray([v.loc for v in self.veh_list], dtype=int),
        )

    def _active_vehicle_by_zone(self):
        counts = np.zeros(self.num_zone, dtype=int)
        for v in self.veh_list:
            if v.state != 2:
                counts[v.loc] += 1
        return counts

    def _vehicle_state_counts(self):
        counts = {"idle_vehicle_count": 0, "busy_vehicle_count": 0,
                  "cruising_vehicle_count": 0, "offline_vehicle_count": 0}
        for v in self.veh_list:
            if v.state == 0:
                counts["idle_vehicle_count"] += 1
            elif v.state == 2:
                counts["offline_vehicle_count"] += 1
            elif v.trace_movement_type == "service":
                counts["busy_vehicle_count"] += 1
            else:
                counts["cruising_vehicle_count"] += 1
        return counts

    def initialize_veh(self):
        veh_locs = np.array([i for i in range(self.num_zone)] * (len(self.veh_profiles) // self.num_zone) + [i for i in
                                                                                                             range(
                                                                                                                 len(self.veh_profiles) - len(
                                                                                                                     self.veh_profiles) // self.num_zone * self.num_zone)])  # np.random.choice(np.arange(self.num_zone), size = len(self.veh_profiles)) #initial state is constant and given (assume we know the initial distribution)
        vid = 0
        for v_profile, v_loc in zip(self.veh_profiles, veh_locs):
            v = Vehicle(vid, v_profile, v_loc)
            if self.use_driver_online_br:
                is_active = self.driver_rng.random() < self.driver_online_probability(v_profile)
            else:
                is_active = v_profile <= self.avg_profit
            if is_active:
                self.veh_count[v_loc] += 1
                self.veh_queue[v_loc].append(v)
                self.active_veh += 1
            else:
                v.state = 2
            self.veh_list.append(v)
            vid += 1

    def get_ongoing_veh(self, max_step=6):
        ongoing_veh_count = np.zeros((max_step, self.num_zone), dtype=int)
        for v in self.veh_list:
            if v.state == 1:
                ongoing_veh_count[min(max_step - 1, int(v.remaining_time) // self.frequency), v.loc] += 1
        return ongoing_veh_count

    def step(self, new_demand, veh_schedule, price_multiplier, compensation_multiplier=None):
        price_multiplier = validate_control(price_multiplier, self.num_zone, "price")
        compensation_multiplier = validate_control(
            price_multiplier if compensation_multiplier is None else compensation_multiplier,
            self.num_zone, "compensation")
        passenger_context_before = (self._passenger_decision_context(price_multiplier)
                                    if self._pending_passenger_audit is None else None)
        #### at time t
        # dispatch_veh
        waiting_request_tensor_before = self.pass_count.copy()
        pricing_history_before = self.pricing_multipliers.copy()
        compensation_history_before = self.compensation_multipliers.copy()
        idle_supply_before = self.veh_count.copy()
        active_vehicle_before = self._active_vehicle_by_zone()
        vehicle_state_before, vehicle_zone_before = self._vehicle_vectors()
        tot_profit = 0
        compensation_basis_total = 0.0
        driver_payment_total = 0.0
        pickup_cost_total = 0.0
        negative_margin_order_count = 0
        negative_margin_total = 0.0
        driver_payment_by_origin = np.zeros(self.num_zone)
        tot_expense = np.zeros(self.num_zone)
        tot_count = np.zeros(self.num_zone)
        occu_mile = 0
        occu_minute = 0
        empty_mile = 0
        empty_minute = 0
        served_pass = 0
        matched_order_count = 0
        matched_matrix = np.zeros((self.num_zone, self.num_zone), dtype=int)
        service_vehicle_origins = []
        service_passenger_origins = []
        service_destinations = []
        service_wait_indices = []
        order_events = []
        vehicle_events = []

        # time step t
        for v_loc, p_loc in veh_schedule:
            p_loc2 = -1
            for j in range(self.max_waiting - 1, -1, -1):
                for i in range(self.num_zone):
                    if self.pass_count[j, p_loc, i] > 0:
                        self.pass_count[j, p_loc, i] -= 1
                        p_loc2 = i
                        break
                if p_loc2 != -1:
                    break
            assert self.veh_count[v_loc] > 0 and p_loc2 != -1, print(
                "Error" + "," + str(np.sum(self.pass_count, axis=(0, 2))) + "," + str(p_loc) + "," + str(v_loc))
            matched_order_count += 1
            order_id = self._next_trace_order_id
            self._next_trace_order_id += 1
            served_pass += 1
            matched_matrix[p_loc, p_loc2] += 1
            v = self.veh_queue[v_loc].popleft()
            pickup_time = float(self.travel_time[v_loc, p_loc])
            trip_time = float(self.travel_time[p_loc, p_loc2])
            v.loc = p_loc2
            v.remaining_time = self.travel_time[v_loc, p_loc] + self.travel_time[p_loc, p_loc2]
            v.state = 1
            v.waiting = 0
            v.trace_from_zone = v_loc
            v.trace_to_zone = p_loc2
            v.trace_total_time = v.remaining_time
            v.trace_movement_type = "service"
            self.veh_count[v_loc] -= 1
            # price calculation function is here
            base_fare = (self.alpha0 + self.alpha1 * self.travel_distance[p_loc, p_loc2]
                         + self.beta1 * self.travel_time[p_loc, p_loc2])
            tmp_payment = self.pricing_multipliers[j, p_loc] * base_fare
            compensation_basis = self.compensation_multipliers[j, p_loc] * base_fare
            driver_payment = self.delta * compensation_basis
            pickup_cost = (self.alpha2 * self.travel_distance[v_loc, p_loc]
                           + self.beta2 * self.travel_time[v_loc, p_loc])
            order_margin = tmp_payment - driver_payment - pickup_cost
            compensation_basis_total += compensation_basis
            driver_payment_total += driver_payment
            driver_payment_by_origin[p_loc] += driver_payment
            pickup_cost_total += pickup_cost
            negative_margin_order_count += int(order_margin < 0)
            negative_margin_total += max(-order_margin, 0.0)
            order_events.append({
                "order_id": int(order_id),
                "vehicle_id": int(v.id),
                "event_type": "served",
                "vehicle_origin": int(v_loc),
                "passenger_origin": int(p_loc),
                "passenger_dest": int(p_loc2),
                "wait_index": int(j),
                "price_multiplier": float(self.pricing_multipliers[j, p_loc]),
                "fare": float(tmp_payment),
                "compensation_multiplier": float(self.compensation_multipliers[j, p_loc]),
                "base_fare": float(base_fare),
                "driver_order_payment": float(driver_payment),
                "pickup_cost": float(pickup_cost),
                "order_margin": float(order_margin),
                "pickup_time": pickup_time,
                "trip_time": trip_time,
                "duration": float(v.remaining_time),
            })
            service_vehicle_origins.append(int(v_loc))
            service_passenger_origins.append(int(p_loc))
            service_destinations.append(int(p_loc2))
            service_wait_indices.append(int(j))
            vehicle_events.append({
                "vehicle_id": int(v.id),
                "event_type": "service",
                "from_zone": int(v_loc),
                "pickup_zone": int(p_loc),
                "to_zone": int(p_loc2),
                "pickup_time": pickup_time,
                "trip_time": trip_time,
                "duration": float(v.remaining_time),
            })
            tot_profit += tmp_payment
            # for the pickup trip, the platform pays drivers according to the empty trip wage
            tot_expense[v.loc] += pickup_cost + driver_payment

            occu_mile += self.travel_distance[p_loc, p_loc2]
            occu_minute += self.travel_time[p_loc, p_loc2]
            empty_mile += self.travel_distance[v_loc, p_loc]
            empty_minute += self.travel_time[v_loc, p_loc]
            tot_count[v.loc] += 1
        # update total and zone profit
        self.avg_profit -= self.avg_profits[0] / 30
        self.avg_profits[:-1] = self.avg_profits[1:]
        income_basis = compensation_basis_total if self.action_mode != "price_only" else tot_profit
        income_denominator = np.sum(self.veh_count) + served_pass + 1e-4
        signal_basis = income_basis
        if self.driver_signal == "net":
            # idle costs in the units of the signal (pay / delta): cruising of this step and, because reposition
            # moves are only drawn later in the step, the reposition cost of the previous step
            signal_basis = income_basis - (self.beta2 * float(np.sum(self.veh_count)) + self._last_reposition_expense) / self.delta
        self.avg_profits[-1] = signal_basis / income_denominator
        self.avg_profit += self.avg_profits[-1] / 30
        self.zone_profit -= self.zone_profits[0, :] / 30
        self.zone_profits[:-1, :] = self.zone_profits[1:, :]
        self.zone_profits[-1, :] = tot_expense / (self.veh_count + tot_count + 1e-4) # wage per avail vehicle in the corresponding zone
        self.zone_profit += self.zone_profits[-1, :] / 30

        # generate reposition destination, implement using a uniform double
        reposition_probabilities = [[] for i in range(self.num_zone)]
        for i in range(self.num_zone):
            if self.veh_count[i] > 0:
                reposition_probabilities[i] = (self.zone_profit + 1e-4) / (self.travel_distance[i, :] + 1e-4)
                reposition_probabilities[i][i] = 0
                reposition_probabilities[i] /= np.sum(reposition_probabilities[i])
                reposition_probabilities[i] = np.cumsum(reposition_probabilities[i])
                # print(reposition_probability)
                # repos_dests[i] = list(np.random.choice(range(self.num_zone), p = reposition_probability/np.sum(reposition_probability) , size=self.veh_count[i]))

        # vehicle cruising within its original zone
        service_expense = float(np.sum(tot_expense))
        idle_cruising_vehicle_count = int(np.sum(self.veh_count))
        cruising_expense = float(self.beta2 * idle_cruising_vehicle_count)
        reposition_expense = 0.0
        tot_expense = service_expense + (cruising_expense if self.idle_cost_bearer == "platform" else 0.0)
        empty_minute += np.sum(self.veh_count)

        # update price_multiplier
        self.pricing_multipliers[1:self.max_waiting, :] = self.pricing_multipliers[0:(self.max_waiting - 1), :]
        self.pricing_multipliers[0, :] = price_multiplier
        self.compensation_multipliers[1:, :] = self.compensation_multipliers[:-1, :].copy()
        self.compensation_multipliers[0, :] = compensation_multiplier

        #### between time t and t+1
        # update_veh
        driver_probability = np.full(len(self.veh_list), np.nan, dtype=float)
        driver_outcome = np.full(len(self.veh_list), np.nan, dtype=float)
        avg_profit_for_driver_decisions = float(self.avg_profit)
        reposition_origins = []
        reposition_destinations = []
        switch_to_offline = 0
        switch_to_online = 0
        isolated = self.env_rng_mode == "isolated"
        if isolated:
            # fixed number of draws per step, indexed by vehicle id, so that streams stay aligned
            # across counterfactual arms whose vehicle states differ
            move_u = self.env_rng.random((3, len(self.veh_list)))
            decision_u = self.driver_rng.random(len(self.veh_list)) if self.use_driver_online_br else None
        for v in self.veh_list:
            ## vehicle movement
            if v.state == 1:
                v.remaining_time -= 1
                if v.remaining_time <= 0:
                    # v arrival
                    v.state = 0
                    v.trace_from_zone = v.loc
                    v.trace_to_zone = v.loc
                    v.trace_total_time = 0
                    v.trace_movement_type = "idle"
                    self.veh_count[v.loc] += 1
                    self.veh_queue[v.loc].append(v)
                    vehicle_events.append({
                        "vehicle_id": int(v.id),
                        "event_type": "arrive_available",
                        "zone": int(v.loc),
                    })
            ## vehicles quit/reposition/rejoin
            elif v.state == 0:
                if not self._driver_decides(v):
                    stays_online = True
                else:
                    if self.use_driver_online_br:
                        online_probability = self.driver_online_probability(v.profile, self.driver_switch_cost)
                        draw = decision_u[v.id] if isolated else self.driver_rng.random()
                        stays_online = draw < online_probability
                    else:
                        stays_online = self._hard_online(v.profile, self.driver_switch_cost)
                        online_probability = float(stays_online)
                    driver_probability[v.id] = online_probability
                    driver_outcome[v.id] = float(stays_online)
                if not stays_online:
                    switch_to_offline += 1
                    old_loc = v.loc
                    v.state = 2
                    v.trace_from_zone = old_loc
                    v.trace_to_zone = old_loc
                    v.trace_total_time = 0
                    v.trace_movement_type = "offline"
                    self.veh_count[v.loc] -= 1
                    self.veh_queue[v.loc].remove(v)
                    self.active_veh -= 1
                    vehicle_events.append({
                        "vehicle_id": int(v.id),
                        "event_type": "offline",
                        "zone": int(old_loc),
                    })
                else:
                    old_loc = v.loc
                    if isolated:
                        new_loc, remaining_time = v.reposition(self.travel_time, move_u[0, v.id],
                                                               (move_u[1, v.id] < reposition_probabilities[v.loc]).argmax())
                    else:
                        if len(self.rand_double) == 0:
                            self.rand_double = list(np.random.random(size=100000))
                        new_loc, remaining_time = v.reposition(self.travel_time, self.rand_double.pop(),\
                                                               (self.rand_double.pop() < reposition_probabilities[v.loc]).argmax())
                    if v.loc != new_loc:
                        move_expense = self.alpha2 * self.travel_distance[v.loc, new_loc] + self.beta2 * self.travel_time[v.loc, new_loc]
                        if self.idle_cost_bearer == "platform":
                            tot_expense += move_expense
                        reposition_expense += float(move_expense)
                        empty_mile += self.travel_distance[v.loc, new_loc]
                        empty_minute += self.travel_time[v.loc, new_loc]
                        self.veh_count[v.loc] -= 1
                        self.veh_queue[v.loc].remove(v)
                        v.state = 1
                        v.loc = new_loc
                        v.remaining_time = remaining_time
                        v.trace_from_zone = old_loc
                        v.trace_to_zone = new_loc
                        v.trace_total_time = remaining_time
                        v.trace_movement_type = "reposition"
                        reposition_origins.append(int(old_loc))
                        reposition_destinations.append(int(new_loc))
                        vehicle_events.append({
                            "vehicle_id": int(v.id),
                            "event_type": "reposition",
                            "from_zone": int(old_loc),
                            "to_zone": int(new_loc),
                            "duration": float(remaining_time),
                        })
            elif v.state == 2:
                if not self._driver_decides(v):
                    goes_online = False
                else:
                    if self.use_driver_online_br:
                        online_probability = self.driver_online_probability(v.profile, -self.driver_switch_cost)
                        draw = decision_u[v.id] if isolated else self.driver_rng.random()
                        goes_online = draw < online_probability
                    else:
                        goes_online = self._hard_online(v.profile, -self.driver_switch_cost)
                        online_probability = float(goes_online)
                    driver_probability[v.id] = online_probability
                    driver_outcome[v.id] = float(goes_online)
                if goes_online:
                    switch_to_online += 1
                    v.state = 0
                    v.waiting = 0
                    v.loc = self._choose_online_zone(move_u[2, v.id] if isolated else None)
                    v.trace_from_zone = v.loc
                    v.trace_to_zone = v.loc
                    v.trace_total_time = 0
                    v.trace_movement_type = "online"
                    if not isolated and len(self.rand_int) == 0:
                        self.rand_int = list(np.random.randint(0, self.num_zone, size = 100000))
                    self.veh_count[v.loc] += 1
                    self.veh_queue[v.loc].append(v)
                    self.active_veh += 1
                    vehicle_events.append({
                        "vehicle_id": int(v.id),
                        "event_type": "online",
                        "zone": int(v.loc),
                    })

        # generate_pass
        left_by_origin = np.sum(self.pass_count[-1, :, :], axis=1).astype(int)
        left_pass = np.sum(left_by_origin)
        self.pass_count[1:, :] = self.pass_count[0:-1, :]
        self.pass_count[0, :] = new_demand

        passenger_stats = self._pending_passenger_stats
        passenger_audit = self._pending_passenger_audit
        if passenger_stats is None:
            new_demand_count = float(np.sum(new_demand))
            passenger_stats = {
                "potential_passenger_count": new_demand_count,
                "actual_passenger_count": new_demand_count,
                "passenger_enter_rate": 1.0 if new_demand_count > 0 else 0.0,
                "passenger_balk_count": 0.0,
            }
            context = passenger_context_before
            passenger_audit = {
                **context,
                "passenger_enter_probability": np.ones(self.num_zone, dtype=float),
                "potential_demand_matrix": np.asarray(new_demand, dtype=int).copy(),
                "actual_demand_matrix": np.asarray(new_demand, dtype=int).copy(),
            }
        self._pending_passenger_stats = None
        self._pending_passenger_audit = None

        potential_driver_count = float(len(self.veh_profiles))
        finite_driver_decisions = ~np.isnan(driver_probability)
        state_counts = self._vehicle_state_counts()
        vehicle_state_after, vehicle_zone_after = self._vehicle_vectors()
        self.last_step_stats = self._empty_step_stats()
        self.last_step_stats.update(passenger_stats)
        self.last_step_stats.update({
            "potential_driver_count": potential_driver_count,
            "active_driver_count": float(self.active_veh),
            "driver_online_rate": float(self.active_veh) / (potential_driver_count + 1e-8),
            "matched_order_count": float(matched_order_count),
            "served_order_count": float(served_pass),
            "driver_decision_count": float(np.sum(finite_driver_decisions)),
            "driver_expected_online_decision_count": float(np.nansum(driver_probability)),
            "driver_actual_online_decision_count": float(np.nansum(driver_outcome)),
            "driver_online_decision_variance": float(
                np.nansum(driver_probability * (1.0 - driver_probability))
            ),
        })
        self.last_step_audit = {
            **passenger_audit,
            **state_counts,
            "passenger_entry_br_enabled": bool(self.use_passenger_entry_br),
            "driver_online_br_enabled": bool(self.use_driver_online_br),
            "price_vector": np.asarray(price_multiplier, dtype=float).copy(),
            "pricing_history_before": pricing_history_before,
            "compensation_history_before": compensation_history_before,
            "compensation_vector": compensation_multiplier.copy(),
            "driver_payment_total": float(driver_payment_total),
            "driver_payment_by_origin": driver_payment_by_origin,
            "pickup_cost_total": float(pickup_cost_total),
            "negative_margin_order_count": int(negative_margin_order_count),
            "negative_margin_total": float(negative_margin_total),
            "driver_switch_count": float(switch_to_offline + switch_to_online),
            "driver_switch_to_offline_count": float(switch_to_offline),
            "driver_switch_to_online_count": float(switch_to_online),
            "driver_income_basis": float(income_basis),
            **self._welfare_audit(passenger_audit, tot_profit, driver_payment_total,
                                  (cruising_expense + reposition_expense) if self.idle_cost_bearer == "driver" else 0.0),
            "driver_income_denominator": float(income_denominator),
            "driver_income_window": np.asarray(self.avg_profits, dtype=float).copy(),
            "joint_action_enabled": self.action_mode != "price_only",
            "waiting_request_tensor_before": waiting_request_tensor_before,
            "waiting_request_tensor_after": self.pass_count.copy(),
            "idle_supply_vector": idle_supply_before,
            "active_vehicle_vector": active_vehicle_before,
            "left_request_from_zone": left_by_origin,
            "matched_matrix": matched_matrix,
            "served_matrix": matched_matrix.copy(),
            "vehicle_state_vector_before": vehicle_state_before,
            "vehicle_zone_vector_before": vehicle_zone_before,
            "vehicle_state_vector_after": vehicle_state_after,
            "vehicle_zone_vector_after": vehicle_zone_after,
            "driver_profile_vector": np.asarray(self.veh_profiles, dtype=float).copy(),
            "driver_decision_probability_vector": driver_probability,
            "driver_decision_outcome_vector": driver_outcome,
            "avg_profit_for_driver_decisions": avg_profit_for_driver_decisions,
            "service_vehicle_origin": np.asarray(service_vehicle_origins, dtype=int),
            "service_passenger_origin": np.asarray(service_passenger_origins, dtype=int),
            "service_destination": np.asarray(service_destinations, dtype=int),
            "service_wait_index": np.asarray(service_wait_indices, dtype=int),
            "reposition_origin": np.asarray(reposition_origins, dtype=int),
            "reposition_destination": np.asarray(reposition_destinations, dtype=int),
            "idle_cruising_vehicle_count": idle_cruising_vehicle_count,
            "travel_distance_matrix": self.travel_distance.copy(),
            "travel_time_matrix": self.travel_time.copy(),
            "profit": float(tot_profit),
            "expense": float(tot_expense),
            "reward": float(tot_profit - tot_expense),
            "service_expense": service_expense,
            "cruising_expense": cruising_expense,
            "reposition_expense": reposition_expense,
        }
        self.last_step_trace = {
            "order_events": order_events,
            "vehicle_events": vehicle_events,
        }

        self._last_reposition_expense = float(reposition_expense)
        self.step_index += 1
        return tot_profit, tot_expense, served_pass, left_pass, occu_mile, occu_minute, empty_mile, empty_minute

    def snapshot_state(self):
        """Deep copy of the complete simulator state (vehicles, queues, histories and generators)."""
        return copy.deepcopy(self.__dict__)

    def restore_state(self, state):
        restored = copy.deepcopy(state)
        self.__dict__.clear()
        self.__dict__.update(restored)

    def reseed_streams(self, seed, step):
        """Give every random stream a state that depends only on (seed, step).

        Counterfactual arms that start from the same snapshot then see identical random numbers at
        every minute regardless of how their vehicle states diverged.  Isolated mode only.
        """
        if self.env_rng_mode != "isolated":
            raise ValueError("reseed_streams requires env_rng='isolated'")
        children = np.random.SeedSequence([int(seed), int(step)]).spawn(3)
        self.passenger_rng, self.driver_rng, self.env_rng = [np.random.default_rng(c) for c in children]

    def passenger_consumer_surplus(self, utility, potential_by_origin):
        """Logit inclusive-value surplus beta * sum_i M_i * log(1 + exp(U_i / beta)).

        The same formula is used whether or not the passenger-entry BR switch is on: it
        always evaluates the configured entry-utility parameters (a measuring convention,
        not a statement that passengers behave stochastically).  M_i is the potential demand
        from origin i, U_i the entry utility given the posted price and the congestion proxy.
        """
        utility = np.asarray(utility, dtype=float)
        potential = np.asarray(potential_by_origin, dtype=float)
        beta = self.beta_passenger
        if beta <= 1e-8:
            return float(np.sum(potential * np.maximum(utility, 0.0)))
        return float(beta * np.sum(potential * np.logaddexp(0.0, utility / beta)))

    def _welfare_audit(self, passenger_audit, revenue, driver_payment_total, driver_idle_cost=0.0):
        potential_by_origin = np.sum(passenger_audit["potential_demand_matrix"], axis=1)
        online_minutes = float(self.active_veh)  # one step is one minute; all non-offline vehicles
        return {
            "consumer_surplus": self.passenger_consumer_surplus(passenger_audit["passenger_utility"], potential_by_origin),
            "passenger_revenue_total": float(revenue),
            "driver_total_income": float(driver_payment_total),
            "online_vehicle_minutes": online_minutes,
            "driver_idle_cost": float(driver_idle_cost),
            "driver_hourly_wage": (float(driver_payment_total) - float(driver_idle_cost)) / online_minutes * 60.0
            if online_minutes > 0 else float("nan"),
            "platform_take_rate": 1.0 - float(driver_payment_total) / float(revenue) if revenue > 0 else float("nan"),
        }

    def reset(self):
        self._last_reposition_expense = 0.0
        self.pass_count *= 0
        # No queued orders survive reset. Match the legacy price-history reset
        # convention, and start compensation history tied to that same history.
        self.compensation_multipliers = self.pricing_multipliers.copy()
        self.veh_count *= 0
        self.veh_queue = [deque() for i in range(self.num_zone)]  # queue for waiting vehicles
        self.veh_list = []  # list of all vehicles
        self.avg_profit = np.mean(self.veh_profiles)
        self.avg_profits = [np.mean(
            self.veh_profiles)] * 30  # active or not depends on the average profit, default_profit is average veh_profit
        self.zone_profits = np.ones((30, self.num_zone))
        self.zone_profit = np.ones(self.num_zone)
        self.active_veh = 0
        self._pending_passenger_stats = None
        self._pending_passenger_audit = None
        self.last_step_stats = self._empty_step_stats()
        self.last_step_audit = {}
        self.last_step_trace = self._empty_step_trace()
        self._next_trace_order_id = 0
        self.initialize_veh()


class Vehicle:
    def __init__(self, vid, profile, zone_id):
        self.id = vid
        self.loc = zone_id
        self.profile = profile
        self.remaining_time = 0
        self.state = 0  # 0 for waiting in a zone, 1 for ongoing, 2 for rest
        self.waiting = 0
        self.trace_from_zone = zone_id
        self.trace_to_zone = zone_id
        self.trace_total_time = 0
        self.trace_movement_type = "idle"

    def reposition(self, travel_time, prob, repos_dest):
        if prob < 0.90951:  # 0.1*np.exp(-0.1), mean as 10 minutes
            self.waiting += 1
            return self.loc, 0
        else:
            self.waiting = 0
            new_loc = repos_dest
            assert self.loc != new_loc
            remaining_time = travel_time[self.loc, new_loc]
        return new_loc, remaining_time
