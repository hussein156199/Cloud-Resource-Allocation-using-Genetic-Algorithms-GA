# ga.py
import random
import copy

def initialize_population(pop_size, num_tasks, num_vms):
    return [[random.randint(0, num_vms - 1) for _ in range(num_tasks)] for _ in range(pop_size)]

def evaluate_fitness(chromosome, tasks, vms, servers, penalty=100.0):
    """
    Compute fitness: lower is better.
    Fitness = makespan (max VM processing time based on actual resource constraints)
            + total cost (task + VM cost)
            + penalties for resource violations
    """
    # Reset VM task lists
    for vm in vms:
        vm.tasks = []

    # Assign tasks to VMs according to chromosome
    for t_idx, vm_idx in enumerate(chromosome):
        if 0 <= vm_idx < len(vms):
            vm = vms[vm_idx]
            task = tasks[t_idx]
            vm.tasks.append(task)

    vm_exec_time = []
    total_cost = 0.0
    penalties = 0.0

    for vm in vms:
        tasks_on_vm = vm.tasks
        if not tasks_on_vm:
            vm_exec_time.append(0.0)
            continue

        # Calculate cost for each task
        for task in tasks_on_vm:
            total_cost += task.cost + (vm.cost * task.time)

            # Immediate resource violation
            if task.cpu > vm.cpu:
                penalties += (task.cpu - vm.cpu) * penalty
            if task.ram > vm.ram:
                penalties += (task.ram - vm.ram) * penalty
            if task.storage > vm.storage:
                penalties += (task.storage - vm.storage) * (penalty / 2)

        # Check if all tasks can execute in parallel on this VM
        cpu_needed = sum(t.cpu for t in tasks_on_vm)
        ram_needed = sum(t.ram for t in tasks_on_vm)
        storage_needed = sum(t.storage for t in tasks_on_vm)

        if cpu_needed <= vm.cpu and ram_needed <= vm.ram and storage_needed <= vm.storage:
            # Can run all in parallel
            vm_time = max(t.time for t in tasks_on_vm)
        else:
            # Must run sequentially
            vm_time = sum(t.time for t in tasks_on_vm)

        vm_exec_time.append(vm_time)

    # Server-level checks: sum VM resources per server
    server_load = {s.id: {"cpu": 0, "ram": 0, "storage": 0} for s in servers}
    for vm in vms:
        server_load[vm.server_id]["cpu"] += vm.cpu
        server_load[vm.server_id]["ram"] += vm.ram
        server_load[vm.server_id]["storage"] += vm.storage

    for s in servers:
        load = server_load[s.id]
        if load["cpu"] > s.cpu:
            penalties += (load["cpu"] - s.cpu) * penalty
        if load["ram"] > s.ram:
            penalties += (load["ram"] - s.ram) * penalty
        if load["storage"] > s.storage:
            penalties += (load["storage"] - s.storage) * (penalty / 2)

    makespan = max(vm_exec_time) if vm_exec_time else 0.0
    fitness = makespan + total_cost + penalties
    return fitness

def tournament_selection(population, fitnesses, k=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants = random.sample(range(pop_size), k)
        best = min(participants, key=lambda i: fitnesses[i])
        selected.append(copy.deepcopy(population[best]))
    return selected

def crossover(parent1, parent2, points=1):
    """multi-point crossover: points = number of cut points"""
    size = len(parent1)
    if size < 2 or points <= 0:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    points = min(points, size - 1)
    cuts = sorted(random.sample(range(1, size), points))
    child1 = parent1[:]
    child2 = parent2[:]
    for i in range(len(cuts)):
        start = cuts[i]
        end = cuts[i+1] if i+1 < len(cuts) else size
        child1[start:end], child2[start:end] = child2[start:end], child1[start:end]
    return child1, child2

def mutate(chromosome, num_vms, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, num_vms - 1)
    return chromosome
