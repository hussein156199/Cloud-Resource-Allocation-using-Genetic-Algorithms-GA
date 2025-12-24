# models.py
class Server:
    def __init__(self, id, cpu, ram, storage, cost=0.0):
        self.id = int(id)
        self.cpu = int(cpu)
        self.ram = int(ram)
        self.storage = int(storage)
        self.cost = int(cost)
        self.vms = []  # list of VM objects assigned to this server

    def remaining(self):
        used_cpu = sum(vm.cpu for vm in self.vms)
        used_ram = sum(vm.ram for vm in self.vms)
        used_storage = sum(vm.storage for vm in self.vms)
        return {"cpu": self.cpu - used_cpu, "ram": self.ram - used_ram, "storage": self.storage - used_storage}

    def remaining_cpu(self):
        used = sum(vm.cpu for vm in self.vms)
        return self.cpu - used

    def remaining_ram(self):
        used = sum(vm.ram for vm in self.vms)
        return self.ram - used

    def remaining_storage(self):
        used = sum(vm.storage for vm in self.vms)
        return self.storage - used

    # def can_add_vm(self, cpu, ram, storage):
    #     return cpu <= self.remaining_cpu() and ram <= self.remaining_ram() and storage <= self.remaining_storage()

    def total_vm_cost(self):
        """Total cost of all VMs on this server."""
        return sum(vm.cost for vm in self.vms)

    def can_add_vm(self, cpu, ram, storage, cost):
        """Check resource and cost capacity."""
        enough_cpu = cpu <= self.remaining_cpu()
        enough_ram = ram <= self.remaining_ram()
        enough_storage = storage <= self.remaining_storage()
        enough_cost = (self.total_vm_cost() + cost) <= self.cost

        return enough_cpu and enough_ram and enough_storage and enough_cost
class VM:
    def __init__(self, id, server_id, cpu, ram, storage, cost=0.0):
        self.id = int(id)
        self.server_id = int(server_id)
        self.cpu = int(cpu)
        self.ram = int(ram)
        self.storage = int(storage)
        self.cost = int(cost)
        self.tasks = []  # tasks assigned (fill when evaluating an allocation)

    def remaining_cpu(self):
        used = sum(task.cpu for task in self.tasks)
        return self.cpu - used

    def remaining_ram(self):
        used = sum(task.ram for task in self.tasks)
        return self.ram - used

    def remaining_storage(self):
        used = sum(task.storage for task in self.tasks)
        return self.storage - used


class Task:
    def __init__(self, id, cpu, ram, storage, time, cost=0.0):
        self.id = int(id)
        self.cpu = int(cpu)
        self.ram = int(ram)
        self.storage = int(storage)
        self.time = float(time)
        self.cost = int(cost)




