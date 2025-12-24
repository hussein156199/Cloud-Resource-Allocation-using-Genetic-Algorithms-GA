# utils.py
import random
from models import Server, VM, Task

def generate_random_data(num_servers, num_vms, num_tasks):
    servers, vms, tasks = [], [], []

    # servers
    for i in range(num_servers):
        cpu = random.randint(20, 64)
        ram = random.randint(32, 256)
        storage = random.randint(500, 2000)
        # cost = round(random.uniform(0.5, 3.0), 2)
        cost = round(random.uniform(50, 200), 2)

        servers.append(Server(i, cpu, ram, storage, cost))

    # VMs — try to place on servers respecting capacity
    vm_id = 0
    attempts = 0
    while vm_id < num_vms and attempts < num_vms * 10:
        s = random.choice(servers)
        cpu = random.randint(1, 12)
        ram = random.randint(2, 32)
        storage = random.randint(20, 200)
        # cost = round(random.uniform(0.1, 1.5), 2)
        cost = round(random.uniform(5, 50), 2)
        if s.can_add_vm(cpu, ram, storage , cost):
            vm = VM(vm_id, s.id, cpu, ram, storage, cost)
            s.vms.append(vm)
            vms.append(vm)
            vm_id += 1
        attempts += 1

    # tasks
    for i in range(num_tasks):
        cpu = random.randint(1, 8)
        ram = random.randint(1, 16)
        storage = random.randint(5, 100)
        time_task = random.randint(1, 10)
        # cost = round(random.uniform(0.1, 2.0), 2)
        cost = round(random.uniform(1, 5), 2)
        tasks.append(Task(i, cpu, ram, storage, time_task, cost))

    return servers, vms, tasks
