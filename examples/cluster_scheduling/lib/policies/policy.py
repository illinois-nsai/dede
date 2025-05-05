from ..job_id_pair import JobIdPair
import time
import random
import numpy as np
import cvxpy as cp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


EPS = 1e-6


class Policy:

    def __init__(self, solver=None):
        self._name = None
        self._solver = solver

    @property
    def name(self):
        return self._name

    def scale_factors_array(self, scale_factors, job_ids, m, n):
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]
        return scale_factors_array

    def flatten(self, d, cluster_spec):
        """Converts a 2-level dict to a NumPy array."""

        job_ids = sorted(list(d.keys()))
        if len(job_ids) == 0:
            return None, None
        worker_types = sorted(list(d[job_ids[0]].keys()))
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]
        if len(worker_types) == 0:
            return None, None
        m = []
        for job_id in job_ids:
            m_row = []
            for worker_type in worker_types:
                m_row.append(d[job_id][worker_type])
            m.append(m_row)
        return np.array(m), (job_ids, worker_types)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_ids, worker_types) = index
        d = {}
        for i in range(len(job_ids)):
            d[job_ids[i]] = {}
            for j in range(len(worker_types)):
                d[job_ids[i]][worker_types[j]] = m[i][j]
        return d

    def get_base_constraints(self, x, scale_factors_array):
        """Return base constraints."""
        return [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= self._num_workers,
            cp.sum(x, axis=1) <= 1,
        ]


class PolicyWithPacking(Policy):

    def __init__(self, solver='ECOS'):
        Policy.__init__(self, solver)

    def scale_factors_array(self, scale_factors, job_ids, m, n):
        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            scale_factor = None
            for single_job_id in job_ids[i].singletons():
                if (scale_factor is not None and
                        scale_factor != scale_factors[single_job_id]):
                    scale_factor = 0
                else:
                    scale_factor = scale_factors[single_job_id]
            for j in range(n):
                scale_factors_array[i, j] = scale_factor
        return scale_factors_array

    def flatten(self, d, cluster_spec, priority_weights=None):
        """
        Converts a 2-level dict to a NumPy array.

        Job ID combinations in the input dict are either a tuple or an integer.
        If a tuple, represents a combination run on a GPU concurrently.
        If an integer, represents a single job / application run on the
        GPU.

        Returns a list of each user's throughput matrix and an
        index to reconstruct the allocation as a dict.
        """
        job_ids = sorted(list(d.keys()))
        if len(job_ids) == 0:
            return None, None
        worker_types = sorted(list(d[job_ids[0]].keys()))
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]

        # Stores which indexes in job_ids are relevant for each single job ID.
        relevant_combinations = {}
        single_job_ids = set()
        sorted_single_job_ids = []
        for i, job_id in enumerate(job_ids):
            if not job_id.is_pair():
                single_job_ids.add(job_id)
                sorted_single_job_ids.append(job_id)
                if job_id not in relevant_combinations:
                    relevant_combinations[job_id] = []
                relevant_combinations[job_id].append(i)
            else:
                for single_job_id in job_id.singletons():
                    if single_job_id not in relevant_combinations:
                        relevant_combinations[single_job_id] = []
                    relevant_combinations[single_job_id].append(i)

        if len(worker_types) == 0:
            return None, None

        shape = (len(single_job_ids), len(job_ids), len(worker_types))
        all_m = np.zeros(shape, dtype=np.float32)
        # Compute the throughput matrix for each individual job.
        for i, single_job_id in enumerate(sorted_single_job_ids):
            # Each throughput matrix has dimension
            # (num_app_combinations x num_worker_types).
            for j in relevant_combinations[single_job_id]:
                job_id = job_ids[j]
                for k, worker_type in enumerate(worker_types):
                    # If job ID of interest is not in this job_id_combination,
                    # throughput should be 0.
                    # Otherwise, use the right throughput from the input dict.
                    if job_id in single_job_ids:
                        if job_id == single_job_id:
                            all_m[i][j][k] = d[job_id][worker_type]
                    else:
                        if single_job_id.overlaps_with(job_id):
                            # Find the index of the job of interest in the job
                            # combination tuple.
                            index = job_id.as_tuple().index(single_job_id[0])
                            throughputs = d[job_id][worker_type]
                            all_m[i][j][k] = d[job_id][worker_type][index]
            # Normalize.
            if priority_weights is not None:
                all_m[i] /= priority_weights[single_job_id]
        return all_m, (job_ids, sorted_single_job_ids, worker_types,
                       relevant_combinations)

    def unflatten(self, m, index):
        """Converts a NumPy array to a 2-level dict."""

        (job_id_combinations, single_job_ids, worker_types, _) = index
        d = {}
        for i in range(len(job_id_combinations)):
            d[job_id_combinations[i]] = {}
            for j in range(len(worker_types)):
                d[job_id_combinations[i]][worker_types[j]] = m[i][j]
        return d

    def get_base_constraints(self, x, single_job_ids,
                             scale_factors_array, relevant_combinations):
        """Return base constraints."""
        constraints = [
            x >= 0,
            cp.sum(cp.multiply(
                scale_factors_array, x), axis=0) <= np.array(self._num_workers),
        ]

        # Every job cannot receive a total time share sum greater than 1.0.
        idx = []
        for single_job_id in single_job_ids:
            indexes = relevant_combinations[single_job_id]
            idx += indexes
        index_var = x[idx]
        index_var = cp.reshape(index_var,
                               (len(single_job_ids), int(np.prod(index_var.shape) /
                                                         len(single_job_ids))), order='C')
        constraints.append(cp.sum(index_var, axis=1) <= 1)
        return constraints

    def convert_job_type_allocation(self, allocation, job_id_to_job_type_key):
        """Converts a job-job_type allocation to a job-job allocation."""
        job_ids = sorted(allocation.keys())
        worker_types = sorted(allocation[job_ids[0]].keys())
        job_type_keys = \
            sorted(set([job_id_to_job_type_key[job_id] for job_id in job_ids]))

        # Initialize job_type-job_type allocation.
        job_type_allocation = {}
        for worker_type in worker_types:
            job_type_allocation[worker_type] = {}
            for job_type_key in job_type_keys:
                job_type_allocation[worker_type][job_type_key] = {}
                job_type_allocation_ = \
                    job_type_allocation[worker_type][job_type_key]
                for other_job_type_key in [None] + job_type_keys:
                    job_type_allocation_[other_job_type_key] = 0.0

        # Populate job_type-job_type allocation.
        for worker_type in worker_types:
            for job_id in allocation:
                job_type_key = job_id_to_job_type_key[job_id]
                for other_job_type_key in allocation[job_id][worker_type]:
                    job_type_allocation[worker_type][job_type_key][other_job_type_key] += \
                        allocation[job_id][worker_type][other_job_type_key]

        # Compute job-job allocations using the following formula:
        # x_{i,j} = x_{i, job_type(j)} * x_{j, job_type(i)} /
        #   sum x_{k, job_type(j)} for all k of job_type(i)
        converted_allocation = {}
        for i, job_id in enumerate(job_ids):
            converted_allocation[job_id] = {}
            job_type_key = job_id_to_job_type_key[job_id]
            # Set the isolated allocations.
            for worker_type in worker_types:
                converted_allocation[job_id][worker_type] = \
                    allocation[job_id][worker_type][None]
            # Set the packed allocations.
            for other_job_id in job_ids[i + 1:]:
                other_job_type_key = job_id_to_job_type_key[other_job_id]
                merged_job_id = \
                    JobIdPair(job_id[0], other_job_id[0])
                converted_allocation[merged_job_id] = {}
                for worker_type in worker_types:
                    current_job_type_allocation = \
                        job_type_allocation[worker_type][job_type_key][other_job_type_key]
                    if current_job_type_allocation > 0.0:
                        if job_type_key == other_job_type_key:
                            current_job_type_allocation -= \
                                allocation[job_id][worker_type][job_type_key]
                        converted_allocation[merged_job_id][worker_type] = \
                            (allocation[job_id][worker_type][other_job_type_key] *
                             allocation[other_job_id][worker_type][job_type_key] /
                             current_job_type_allocation)
                    else:
                        converted_allocation[merged_job_id][worker_type] = 0.0

        return converted_allocation


class MaxMinFairnessPolicy(Policy):

    def __init__(self, solver=None):
        self._name = 'MaxMinFairness'
        self._solver = solver
        self._max_min_fairness_perf_policy = \
            MaxMinFairnessPolicyWithPerf(solver)

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            return None
        (job_ids, worker_types) = index

        new_unflattened_throughputs = {}
        for job_id in unflattened_throughputs:
            new_unflattened_throughputs[job_id] = {}
            for worker_type in unflattened_throughputs[job_id]:
                new_unflattened_throughputs[job_id][worker_type] = 1.0

        return self._max_min_fairness_perf_policy.get_allocation(
            new_unflattened_throughputs, scale_factors, priority_weights,
            cluster_spec)


class MaxMinFairnessPolicyWithPerf(Policy):

    def __init__(self, solver=None):
        Policy.__init__(self, solver)
        self._name = 'MaxMinFairness_Perf'
        self._solver = solver
        self._proportional_policy = ProportionalPolicy()

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        proportional_throughputs = self._proportional_policy.get_throughputs(
            throughputs, index, cluster_spec)
        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        x = cp.Variable(throughputs.shape)

        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.min(cp.sum(cp.multiply(
                np.multiply(throughputs * priority_weights.reshape((m, 1)),
                            scale_factors_array), x), axis=1)))
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        kwargs = {}
        result = cvxprob.solve(solver=self._solver, **kwargs)
        print(
            f'Optimal solution solve x={x.shape}, objective: {result:.4f}, solve time: {cvxprob.solver_stats.solve_time:4f}')

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(x.value.clip(min=0.0).clip(max=1.0), index)


class MaxProportionalFairness(Policy):

    def __init__(self, solver=None):
        Policy.__init__(self, solver)
        self._name = 'MaxProportionalFairness'
        self._solver = solver
        self._proportional_policy = ProportionalPolicy()

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index

        # Row i of scale_factors_array is the scale_factor of job i
        # repeated len(worker_types) times.
        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, m, n)

        # nonneg=True is necessary to avoid solver failure
        x = cp.Variable(throughputs.shape, nonneg=True)

        # Multiply throughputs by scale_factors to ensure that scale_factor
        # is taken into account while allocating times to different jobs.
        # A job run on 1 GPU should receive `scale_factor` more time than
        # a job run on `scale_factor` GPUs if throughputs are equal.
        objective = cp.Maximize(
            cp.log(cp.multiply(throughputs * scale_factors_array, x).sum(1) + EPS).sum())
        # Make sure that the allocation can fit in the cluster.
        constraints = self.get_base_constraints(x, scale_factors_array)
        cvxprob = cp.Problem(objective, constraints)
        kwargs = {}
        result = cvxprob.solve(solver=self._solver, **kwargs)

        # fix violation
        fix_x = x.value.clip(min=0.0).clip(max=1.0)
        fix_x = fix_x / np.maximum((scale_factors_array * fix_x).sum(0) /
                                   (np.array(self._num_workers) + EPS), 1)[None, :]
        fix_x = fix_x / np.maximum(fix_x.sum(1), 1)[:, None]
        result = np.log((throughputs * scale_factors_array * fix_x).sum(1) + EPS).sum()
        print(
            f'Optimal solution solve x={x.shape}, objective: {result:.4f}, solve time: {cvxprob.solver_stats.solve_time:4f}')

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')

        return super().unflatten(fix_x, index)


class ProportionalPolicy(Policy):

    def __init__(self):
        self._name = 'Proportional'

    def get_throughputs(self, throughputs, index,
                        cluster_spec):
        if throughputs is None:
            return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        x_proportional = self._get_allocation(
            throughputs, index,
            cluster_spec)
        proportional_throughputs = np.sum(np.multiply(throughputs, x_proportional),
                                          axis=1).reshape((m, 1))
        return proportional_throughputs

    def _get_allocation(self, throughputs, index,
                        cluster_spec):
        (_, worker_types) = index
        (m, n) = throughputs.shape

        # Split cluster over users (m).
        # x[i, j] proportional to num_workers[j].
        # \sum_j x[i, j] <= 1 for all i.
        # \sum_i x[i, j] <= 1 for all j.
        x = np.array([[cluster_spec[worker_type] / m for worker_type in worker_types]
                      for i in range(m)])
        max_per_row_sum = np.sum(x, axis=1).max()
        x = x / max_per_row_sum

        return x

    def get_allocation(self, unflattened_throughputs,
                       cluster_spec):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None:
            return None
        (job_ids, worker_types) = index
        (m, n) = throughputs.shape

        x = self._get_allocation(throughputs, index,
                                 cluster_spec)

        return super().unflatten(x, index)


class GandivaPolicy(PolicyWithPacking):

    def __init__(self, seed=None):
        self._name = 'Gandiva'
        self._assigned_combinations = {}
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)

    def _get_allocation(self, job_combinations_to_schedule, index, scale_factors,
                        cluster_spec):
        # Helper method that divides time equally among all job combinations in
        # job_combinations_to_schedule.

        (job_ids, single_job_ids, worker_types, relevant_combinations) = index
        m = len(job_combinations_to_schedule)

        indices = {}
        for i, job_id in enumerate(job_ids):
            indices[job_id] = i

        job_combination_indices_to_schedule = []
        for job_combination_to_schedule in job_combinations_to_schedule:
            if job_combination_to_schedule in indices:
                job_combination_indices_to_schedule.append(indices[job_combination_to_schedule])

        scale_factors_array = self.scale_factors_array(
            scale_factors, job_ids, len(job_ids), len(worker_types))

        # Split cluster over users (m). By construction,
        # \sum_i (x[i, j] * scale_factor[i]) = num_workers[j].
        # Normalize to ensure \sum_j x[i, j] <= 1 for all i.
        x = np.zeros((len(job_ids), len(worker_types)))
        for i in job_combination_indices_to_schedule:
            x[i] = np.array([cluster_spec[worker_type] / m for worker_type in worker_types])
            x[i] = x[i] / scale_factors_array[i]
        per_row_sum = np.sum(x, axis=1)
        per_row_sum = np.maximum(per_row_sum, np.ones(per_row_sum.shape))
        x = x / per_row_sum[:, None]

        return x

    def _get_normalized_throughput(self, job_combination, throughputs, worker_types):
        normalized_packed_throughput = 0.0
        if not job_combination.is_pair():
            return 0.0
        for worker_type in worker_types:
            if job_combination not in throughputs:
                return 0.0
            packed_throughput = throughputs[job_combination][worker_type]
            for i, single_job_id in enumerate(job_combination.singletons()):
                if packed_throughput[i] <= 0.0:
                    return 0.0
                isolated_throughput = \
                    throughputs[single_job_id][worker_type]
                normalized_packed_throughput += \
                    packed_throughput[i] / isolated_throughput
        return normalized_packed_throughput

    def get_allocation(self, unflattened_throughputs, scale_factors,
                       unflattened_priority_weights, cluster_spec):
        start = time.time()
        all_throughputs, index = super().flatten(unflattened_throughputs,
                                                 cluster_spec)
        if all_throughputs is None or len(all_throughputs) == 0:
            return None

        (m, n) = all_throughputs[0].shape
        (job_ids, single_job_ids, worker_types, relevant_combinations) = index

        assigned_combination_keys = self._assigned_combinations.keys()
        to_delete = []
        for job_id in assigned_combination_keys:
            (job_combination, other_job_id) = self._assigned_combinations[job_id]
            if job_id not in job_ids:
                to_delete.extend([job_id, other_job_id])
                continue
            if other_job_id is not None:
                if other_job_id not in job_ids:
                    to_delete.extend([job_id, other_job_id])
                    continue
            # Stop using combinations with normalized throughput < 1.0.
            if self._get_normalized_throughput(job_combination, unflattened_throughputs,
                                               worker_types) < 1.0:
                to_delete.extend([job_id, other_job_id])
        for job_id in to_delete:
            if job_id is not None:
                if job_id in self._assigned_combinations:
                    del self._assigned_combinations[job_id]

        num_workers_requested = 0
        for single_job_id in single_job_ids:
            num_workers_requested += scale_factors[single_job_id]
        num_workers_available = 0
        for worker_type in worker_types:
            num_workers_available += cluster_spec[worker_type]

        if num_workers_requested <= num_workers_available:
            # If jobs fit in cluster, do not deploy packing.
            x = self._get_allocation(single_job_ids, index, scale_factors,
                                     cluster_spec)
        else:
            # Deploy packing.
            # Assign all job IDs that are not yet in combinations to combinations.
            to_be_assigned_combinations = []
            for single_job_id in single_job_ids:
                if single_job_id not in self._assigned_combinations:
                    to_be_assigned_combinations.append(single_job_id)

            # Randomly group jobs.
            i = 0
            n = len(to_be_assigned_combinations)
            while len(to_be_assigned_combinations) > 1 and i < n:
                i += 1
                [job1_id, job2_id] = self._rng.sample(
                    to_be_assigned_combinations, 2)
                # Make sure scale factors of jobs in job combination are the
                # same.
                if scale_factors[job1_id] != scale_factors[job2_id]:
                    continue
                to_be_assigned_combinations.remove(job1_id)
                to_be_assigned_combinations.remove(job2_id)
                job_combination = JobIdPair(job1_id[0], job2_id[0])
                self._assigned_combinations[job1_id] = (job_combination,
                                                        job2_id)
                self._assigned_combinations[job2_id] = (job_combination,
                                                        job1_id)
            for i in range(len(to_be_assigned_combinations)):
                job_id = to_be_assigned_combinations[i]
                self._assigned_combinations[job_id] = (job_id, None)

            job_combinations_to_schedule = set()
            for single_job_id in self._assigned_combinations:
                job_combinations_to_schedule.add(
                    self._assigned_combinations[single_job_id][0])
            job_combinations_to_schedule = list(job_combinations_to_schedule)

            x = self._get_allocation(job_combinations_to_schedule, index,
                                     scale_factors,
                                     cluster_spec)

        job_ids = sorted(list(unflattened_throughputs.keys()))
        if len(job_ids) == 0:
            return None, None
        worker_types = sorted(list(unflattened_throughputs[job_ids[0]].keys()))
        self._num_workers = \
            [cluster_spec[worker_type] for worker_type in worker_types]
        if len(worker_types) == 0:
            return None, None
        m = []
        for job_id in job_ids:
            m_row = []
            for worker_type in worker_types:
                m_row.append(unflattened_throughputs[job_id][worker_type])
            m.append(m_row)
        throughputs, index_ = np.array(m), (job_ids, worker_types)

        if throughputs is None:
            return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index_

        scale_factors_array = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                scale_factors_array[i, j] = scale_factors[job_ids[i]]

        priority_weights = np.array(
            [1. / unflattened_priority_weights[job_id]
             for job_id in job_ids])

        x = np.array([[cluster_spec[worker_type] / m for worker_type in worker_types]
                      for i in range(m)])
        max_per_row_sum = np.sum(x, axis=1).max()
        x_proportional = x / max_per_row_sum
        proportional_throughputs = np.sum(np.multiply(throughputs, x_proportional),
                                          axis=1).reshape((m, 1))

        priority_weights = np.multiply(priority_weights.reshape((m, 1)),
                                       1.0 / proportional_throughputs.reshape((m, 1)))

        results = np.multiply(
            np.multiply(
                throughputs *
                priority_weights.reshape(
                    (m,
                     1)),
                scale_factors_array),
            x).sum(1)
        results = results[x.sum(1) > 0].min()

        x = super().unflatten(x, index)
        return x
