from collections import deque
import numpy as np
from economic_action import validate_control

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

        self.rand_int = list(np.random.randint(0,self.num_zone,size=100000))
        self.rand_double = list(np.random.random(size=100000))

    def configure_bounded_rationality(self, config=None):
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

        seed = getattr(config, "br_random_seed", None)
        if seed is None:
            seed = getattr(config, "seed", None)
        self.br_random_seed = seed
        seed_sequence = np.random.SeedSequence(seed)
        passenger_seed, driver_seed = seed_sequence.spawn(2)
        self.passenger_rng = np.random.default_rng(passenger_seed)
        self.driver_rng = np.random.default_rng(driver_seed)

        self.beta_passenger = float(getattr(config, "beta_passenger", 1.0))
        self.beta_driver = float(getattr(config, "beta_driver", 1.0))
        self.passenger_base_utility = float(getattr(config, "passenger_base_utility", 2.0))
        self.passenger_price_sensitivity = float(getattr(config, "passenger_price_sensitivity", 1.0))
        self.passenger_wait_sensitivity = float(getattr(config, "passenger_wait_sensitivity", 0.1))

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

    def _passenger_decision_context(self, price_multiplier):
        waiting_by_origin = np.sum(self.pass_count, axis=(0, 2))
        idle_supply = self.veh_count.copy()
        congestion = waiting_by_origin / (idle_supply + 1.0)
        utility = (
            self.passenger_base_utility
            - self.passenger_price_sensitivity * np.asarray(price_multiplier)
            - self.passenger_wait_sensitivity * congestion ** 2
        )
        return {
            "waiting_by_origin": waiting_by_origin.astype(float),
            "idle_supply": idle_supply.astype(float),
            "congestion_proxy": congestion.astype(float),
            "passenger_utility": utility.astype(float),
            "passenger_enter_probability": self._sigmoid_probability(utility, self.beta_passenger),
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

    def driver_online_probability(self, driver_profile):
        utility = self.avg_profit - driver_profile
        return float(self._sigmoid_probability(utility, self.beta_driver))

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
        income_basis = compensation_basis_total if self.action_mode == "joint" else tot_profit
        self.avg_profits[-1] = income_basis / (np.sum(self.veh_count) + served_pass + 1e-4)
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
        tot_expense = service_expense + cruising_expense
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
                if self.use_driver_online_br:
                    online_probability = self.driver_online_probability(v.profile)
                    stays_online = self.driver_rng.random() < online_probability
                else:
                    online_probability = float(v.profile <= self.avg_profit)
                    stays_online = v.profile <= self.avg_profit
                driver_probability[v.id] = online_probability
                driver_outcome[v.id] = float(stays_online)
                if not stays_online:
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
                    if len(self.rand_double) == 0:
                        self.rand_double = list(np.random.random(size=100000))
                    old_loc = v.loc
                    new_loc, remaining_time = v.reposition(self.travel_time, self.rand_double.pop(),\
                                                           (self.rand_double.pop() < reposition_probabilities[v.loc]).argmax())
                    if v.loc != new_loc:
                        move_expense = self.alpha2 * self.travel_distance[v.loc, new_loc] + self.beta2 * self.travel_time[v.loc, new_loc]
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
                if self.use_driver_online_br:
                    online_probability = self.driver_online_probability(v.profile)
                    goes_online = self.driver_rng.random() < online_probability
                else:
                    online_probability = float(v.profile <= self.avg_profit)
                    goes_online = v.profile <= self.avg_profit
                driver_probability[v.id] = online_probability
                driver_outcome[v.id] = float(goes_online)
                if goes_online:
                    v.state = 0
                    v.waiting = 0
                    v.loc = self.rand_int.pop()
                    v.trace_from_zone = v.loc
                    v.trace_to_zone = v.loc
                    v.trace_total_time = 0
                    v.trace_movement_type = "online"
                    if len(self.rand_int) == 0:
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
            context = self._passenger_decision_context(price_multiplier)
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
            "driver_income_basis": float(income_basis),
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

        return tot_profit, tot_expense, served_pass, left_pass, occu_mile, occu_minute, empty_mile, empty_minute

    def reset(self):
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
