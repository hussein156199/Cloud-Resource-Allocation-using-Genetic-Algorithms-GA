import numpy as np
import random
from copy import deepcopy

from matplotlib import pyplot as plt


class DifferentialEvolution:
    def __init__(self, vms, servers ,tasks, pop_size, generations, F, CR):
        self.vms = vms
        self.tasks = tasks
        self.servers = servers if servers else []
        self.pop_size = pop_size
        self.generations = generations
        self.F = F
        self.CR = CR

        self.dim = len(tasks)  # كل مهمة هي بعد في الـ chromosome
        self.vm_count = len(vms)

        # Initialize population: كل مهمة تختار VM عشوائي
        self.population = np.random.randint(0, self.vm_count, (pop_size, self.dim))


    def fitness(self, solution):
        # إعادة تعيين المهام على كل VM
        for vm in self.vms:
            vm.tasks = []

        total_cost = 0.0
        penalties = 0.0
        vm_times = [0.0] * self.vm_count

        # Assign tasks to VMs
        for t_idx, vm_idx in enumerate(solution):
            task = self.tasks[t_idx]
            vm_idx = int(round(vm_idx))
            vm_idx = max(0, min(self.vm_count - 1, vm_idx))
            vm = self.vms[vm_idx]
            vm.tasks.append(task)

            # حساب التكلفة لكل مهمة
            total_cost += task.cost + (vm.cost * task.time)
            vm_times[vm_idx] += task.time

            # العقوبات لو تجاوزت الموارد
            if task.cpu > vm.cpu:
                penalties += (task.cpu - vm.cpu) * 100
            if task.ram > vm.ram:
                penalties += (task.ram - vm.ram) * 100
            if task.storage > vm.storage:
                penalties += (task.storage - vm.storage) * 50

        # حساب makespan لكل VM
        for vm in self.vms:
            tasks_on_vm = vm.tasks
            if tasks_on_vm:
                cpu_needed = sum(t.cpu for t in tasks_on_vm)
                ram_needed = sum(t.ram for t in tasks_on_vm)
                storage_needed = sum(t.storage for t in tasks_on_vm)

                if cpu_needed > vm.cpu or ram_needed > vm.ram or storage_needed > vm.storage:
                    # لو الموارد لا تسمح، نفترض تنفيذ متسلسل
                    vm_times[self.vms.index(vm)] = sum(t.time for t in tasks_on_vm)
                else:
                    # تنفيذ متوازي
                    vm_times[self.vms.index(vm)] = max(t.time for t in tasks_on_vm)

        makespan = max(vm_times) if vm_times else 0.0

        # عقوبات مستوى السيرفر
        server_load = {s.id: {"cpu": 0, "ram": 0, "storage": 0} for s in self.servers}
        for vm in self.vms:
            server_load[vm.server_id]["cpu"] += vm.cpu
            server_load[vm.server_id]["ram"] += vm.ram
            server_load[vm.server_id]["storage"] += vm.storage

        for s in self.servers:
            load = server_load[s.id]
            if load["cpu"] > s.cpu:
                penalties += (load["cpu"] - s.cpu) * 100
            if load["ram"] > s.ram:
                penalties += (load["ram"] - s.ram) * 100
            if load["storage"] > s.storage:
                penalties += (load["storage"] - s.storage) * 50

        # Fitness النهائي
        return total_cost + penalties + makespan
    # ---------------------------------------------------------
    def mutate(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = random.sample(indices, 3)

        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, 0, self.vm_count - 1)
        return mutant

    # ---------------------------------------------------------
    def crossover(self, target, mutant):
        trial = np.copy(target)
        j_rand = random.randint(0, self.dim - 1)
        for j in range(self.dim):
            if random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    # ---------------------------------------------------------
    def select(self, target, trial):
        return trial if self.fitness(trial) < self.fitness(target) else target

    # ---------------------------------------------------------

    def run(self):
        import time
        history = []
        best = None
        first_best_time = None
        start_time = time.time()  # بداية التوقيت

        for gen in range(self.generations):
            new_pop = []
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                selected = self.select(self.population[i], trial)
                new_pop.append(selected)

            self.population = np.array(new_pop)

            # أفضل حل في هذا الجيل
            fitness_values = [self.fitness(ind) for ind in self.population]
            best_idx = np.argmin(fitness_values)
            best_in_gen = deepcopy(self.population[best_idx])
            best_fit_in_gen = fitness_values[best_idx]

            # تحديث أفضل حل مطلق
            if best is None or best_fit_in_gen < self.fitness(best):
                best = best_in_gen
                # سجل وقت أول أفضل حل
                if first_best_time is None:
                    first_best_time = time.time() - start_time

            history.append(best_fit_in_gen)

        return best, self.fitness(best), history, first_best_time
