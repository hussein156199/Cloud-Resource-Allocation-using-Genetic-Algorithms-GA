import streamlit as st
import pandas as pd
import numpy as np
from models import Server, VM, Task
from utils import generate_random_data
from de import DifferentialEvolution
import plotly.express as px
import plotly.graph_objects as go

# ------------------ Session State ------------------
if 'servers' not in st.session_state:
    st.session_state.servers = []
if 'vms' not in st.session_state:
    st.session_state.vms = []
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'ga_all_runs' not in st.session_state:
    st.session_state.ga_all_runs = []
if 'de_all_runs' not in st.session_state:
    st.session_state.de_all_runs = []
if 'overall_best_ga' not in st.session_state:
    st.session_state.overall_best_ga = None
if 'overall_best_de' not in st.session_state:
    st.session_state.overall_best_de = None

# ------------------ GA Generator ------------------
def run_ga_generator(tasks, vms, servers, pop_size=50, generations=50, mutation_rate=0.1, crossover_points=2, elitism=1):
    def fitness(chrom):
        total_cost = 0
        total_time = 0
        for task_idx, vm_idx in enumerate(chrom):
            task = tasks[task_idx]
            vm = vms[vm_idx % len(vms)]
            total_cost += task.cost + vm.cost
            total_time += task.time
        return total_cost + total_time

    population = [np.random.randint(0, len(vms), size=len(tasks)) for _ in range(pop_size)]

    for gen in range(generations):
        fitness_values = np.array([fitness(ind) for ind in population])
        best_idx = np.argmin(fitness_values)
        best_fit = fitness_values[best_idx]
        best_chrom = population[best_idx]

        yield gen, best_chrom, best_fit, population

        new_population = []
        while len(new_population) < pop_size:
            a, b = np.random.choice(pop_size, 2, replace=False)
            parent1 = population[a] if fitness_values[a] < fitness_values[b] else population[b]

            c, d = np.random.choice(pop_size, 2, replace=False)
            parent2 = population[c] if fitness_values[c] < fitness_values[d] else population[d]

            # Crossover
            crossover_points_idx = np.sort(np.random.choice(len(tasks), crossover_points, replace=False))
            child = parent1.copy()
            for idx in crossover_points_idx:
                child[idx] = parent2[idx]

            # Mutation
            for i in range(len(child)):
                if np.random.rand() < mutation_rate:
                    child[i] = np.random.randint(0, len(vms))

            new_population.append(child)

        # Elitism
        top_indices = np.argsort(fitness_values)[:elitism]
        for i in top_indices:
            new_population[i] = population[i].copy()

        population = new_population

# ------------------ GA Runner ------------------
def run_ga(pop, gens, cpoints, mrate, elit, tlimit, num_runs):
    if not st.session_state.tasks or not st.session_state.vms or not st.session_state.servers:
        st.error("Add at least one Server, VM, and Task")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state.ga_all_runs = []
    overall_best_fit = float('inf')
    overall_best_chrom = None

    for run_idx in range(num_runs):
        status_text.text(f"Running GA Run {run_idx + 1}")
        history = []
        gen_iter = run_ga_generator(
            st.session_state.tasks, st.session_state.vms, st.session_state.servers,
            pop_size=pop, generations=gens, mutation_rate=mrate, crossover_points=cpoints, elitism=elit
        )

        for gen_idx, best_chrom, best_fit, _ in gen_iter:
            history.append(best_fit)
            progress_bar.progress((run_idx * gens + gen_idx + 1) / (num_runs * gens))
            status_text.text(f"Run {run_idx + 1}, Gen {gen_idx + 1}, Best Fitness: {best_fit:.4f}")

        st.session_state.ga_all_runs.append(history)
        if min(history) < overall_best_fit:
            overall_best_fit = min(history)
            overall_best_chrom = best_chrom

    st.session_state.overall_best_ga = (overall_best_chrom, overall_best_fit)

    # GA Plot
    fig = go.Figure()
    for i, hist in enumerate(st.session_state.ga_all_runs):
        fig.add_trace(go.Scatter(
            y=hist, x=list(range(1, len(hist) + 1)),
            mode='lines+markers', name=f'Run {i + 1}',
            hovertemplate='Gen %{x}<br>Fitness %{y:.4f}'
        ))
    fig.update_layout(title="GA Fitness Progress", xaxis_title="Generation", yaxis_title="Best Fitness",
                      width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)
    status_text.text("GA completed")

# ------------------ DE Runner ------------------
def run_de(pop, gens, cr, f_weight, num_runs):
    if not st.session_state.tasks or not st.session_state.vms:
        st.error("Load VMs and Tasks first")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state.de_all_runs = []
    best_of_all = None
    best_fit_of_all = float('inf')

    for run_idx in range(num_runs):
        status_text.text(f"Running DE Run {run_idx + 1}")
        de = DifferentialEvolution(vms=st.session_state.vms, tasks=st.session_state.tasks,
                                   servers=st.session_state.servers,
                                   pop_size=pop, generations=gens, F=f_weight, CR=cr)
        history = []

        for gen in range(gens):
            new_pop = []
            for i in range(pop):
                mutant = de.mutate(i)
                trial = de.crossover(de.population[i], mutant)
                selected = de.select(de.population[i], trial)
                new_pop.append(selected)
            de.population = np.array(new_pop)
            fitness_values = [de.fitness(ind) for ind in de.population]
            best_fit = min(fitness_values)
            history.append(best_fit)
            progress_bar.progress((run_idx * gens + gen + 1) / (num_runs * gens))
            status_text.text(f"DE Run {run_idx + 1}, Gen {gen + 1}, Best Fitness: {best_fit:.4f}")

        st.session_state.de_all_runs.append(history)
        if best_fit < best_fit_of_all:
            best_fit_of_all = best_fit
            best_of_all = de.population[np.argmin(fitness_values)]

    st.session_state.overall_best_de = (best_of_all, best_fit_of_all)

    # DE Plot
    fig = go.Figure()
    for i, hist in enumerate(st.session_state.de_all_runs):
        fig.add_trace(go.Scatter(
            y=hist, x=list(range(1, len(hist) + 1)),
            mode='lines+markers', name=f'Run {i + 1}',
            hovertemplate='Gen %{x}<br>Fitness %{y:.4f}'
        ))
    fig.update_layout(title="DE Fitness Progress", xaxis_title="Generation", yaxis_title="Best Fitness",
                      width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)
    status_text.text("DE completed")

# ------------------ Visualization ------------------
def visualize_data():
    if not (st.session_state.servers and st.session_state.vms and st.session_state.tasks):
        st.warning("No data to visualize")
        return

    data = []
    # Servers
    for s in st.session_state.servers:
        data.append({"id": f"Server {s.id}", "parent": "", "label": f"Server {s.id}"})
    # VMs
    for v in st.session_state.vms:
        data.append({"id": f"VM {v.id}", "parent": f"Server {v.server_id}", "label": f"VM {v.id}"})
    # Tasks based on GA best chromosome
    if st.session_state.overall_best_ga:
        chrom, _ = st.session_state.overall_best_ga
        for t_idx, vm_idx in enumerate(chrom):
            task = st.session_state.tasks[t_idx]
            vm = st.session_state.vms[vm_idx % len(st.session_state.vms)]
            data.append({"id": f"Task {task.id}", "parent": f"VM {vm.id}", "label": f"Task {task.id}"})
    else:
        # Fallback: random VM assignment
        for t in st.session_state.tasks:
            vm = np.random.choice(st.session_state.vms)
            data.append({"id": f"Task {t.id}", "parent": f"VM {vm.id}", "label": f"Task {t.id}"})

    df = pd.DataFrame(data)
    fig = px.treemap(df, names='label', ids='id', parents='parent',
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit Interface ------------------
def main():
    st.set_page_config(page_title="Cloud Resource Allocation", layout="wide")
    st.title("☁️ Cloud Resource Allocation - GA & DE")

    tabs = st.tabs([
        "Servers", "VMs", "Tasks", "GA Params", "GA Run", "DE Params", "DE Run",
        "GA Summary", "DE Summary", "Comparison", "Visualization"
    ])

    # Servers Tab
    with tabs[0]:
        st.header("Add Server")
        id_val = st.number_input("Server ID", min_value=0, key="srv_id")
        cpu = st.number_input("CPU", min_value=0, key="srv_cpu")
        ram = st.number_input("RAM", min_value=0, key="srv_ram")
        storage = st.number_input("Storage", min_value=0, key="srv_st")
        cost = st.number_input("Cost", min_value=0.0, key="srv_cost")
        if st.button("Add Server"):
            if any(s.id == id_val for s in st.session_state.servers):
                st.error(f"Server ID {id_val} already exists!")
            else:
                server = Server(id_val, cpu, ram, storage, cost)
                st.session_state.servers.append(server)
                st.success(f"Server {id_val} added")
        st.subheader("Current Servers")
        for s in st.session_state.servers:
            st.info(f"ID: {s.id}, CPU: {s.cpu}, RAM: {s.ram}, Storage: {s.storage}, Cost: {s.cost}")

    # VMs Tab
    with tabs[1]:
        st.header("Add VM")
        if st.session_state.servers:
            id_val = st.number_input("VM ID", min_value=0, key="vm_id")
            server_id = st.selectbox("Server", [s.id for s in st.session_state.servers])
            cpu = st.number_input("CPU", min_value=0, key="vm_cpu")
            ram = st.number_input("RAM", min_value=0, key="vm_ram")
            storage = st.number_input("Storage", min_value=0, key="vm_st")
            cost = st.number_input("Cost", min_value=0.0, key="vm_cost")
            if st.button("Add VM"):
                if any(v.id == id_val for v in st.session_state.vms):
                    st.error(f"VM ID {id_val} already exists!")
                else:
                    vm = VM(id_val, server_id, cpu, ram, storage, cost)
                    st.session_state.vms.append(vm)
                    st.success(f"VM {id_val} added to Server {server_id}")
        st.subheader("Current VMs")
        for v in st.session_state.vms:
            st.info(f"ID: {v.id}, Server: {v.server_id}, CPU: {v.cpu}, RAM: {v.ram}, Storage: {v.storage}, Cost: {v.cost}")

    # Tasks Tab
    with tabs[2]:
        st.header("Add Task")
        id_val = st.number_input("Task ID", min_value=0, key="task_id")
        cpu = st.number_input("CPU", min_value=0, key="task_cpu")
        ram = st.number_input("RAM", min_value=0, key="task_ram")
        storage = st.number_input("Storage", min_value=0, key="task_st")
        time_val = st.number_input("Time", min_value=0.0, key="task_time")
        cost = st.number_input("Cost", min_value=0.0, key="task_cost")
        if st.button("Add Task"):
            if any(t.id == id_val for t in st.session_state.tasks):
                st.error(f"Task ID {id_val} already exists!")
            else:
                task = Task(id_val, cpu, ram, storage, time_val, cost)
                st.session_state.tasks.append(task)
                st.success(f"Task {id_val} added")
        st.subheader("Current Tasks")
        for t in st.session_state.tasks:
            st.info(f"ID: {t.id}, CPU: {t.cpu}, RAM: {t.ram}, Storage: {t.storage}, Time: {t.time}, Cost: {t.cost}")

    # GA Params
    with tabs[3]:
        st.header("GA Parameters")
        pop = st.number_input("Population", min_value=1, value=50)
        gens = st.number_input("Generations", min_value=1, value=50)
        cpoints = st.number_input("Crossover points", min_value=1, value=2)
        mrate = st.slider("Mutation rate", 0.0, 1.0, 0.2)
        elit = st.number_input("Elitism", min_value=0, value=1)
        runs = st.number_input("Number of runs", min_value=1, value=1)

    # GA Run
    with tabs[4]:
        if st.button("Start GA"):
            run_ga(pop, gens, cpoints, mrate, elit, 0, runs)

    # DE Params
    with tabs[5]:
        st.header("DE Parameters")
        de_pop = st.number_input("DE Population", min_value=1, value=50)
        de_gens = st.number_input("DE Generations", min_value=1, value=50)
        cr = st.slider("CR", 0.0, 1.0, 0.9)
        f_weight = st.slider("F", 0.0, 2.0, 0.8)
        de_runs = st.number_input("DE runs", min_value=1, value=1)

    # DE Run
    with tabs[6]:
        if st.button("Start DE"):
            run_de(de_pop, de_gens, cr, f_weight, de_runs)

    # GA Summary
    with tabs[7]:
        st.header("GA Summary")
        if st.session_state.overall_best_ga:
            chrom, fit = st.session_state.overall_best_ga
            st.success(f"Best GA Fitness: {fit:.4f}")
            st.write(f"Best GA Chromosome: {chrom}")

    # DE Summary
    with tabs[8]:
        st.header("DE Summary")
        if st.session_state.overall_best_de:
            chrom, fit = st.session_state.overall_best_de
            st.success(f"Best DE Fitness: {fit:.4f}")
            st.write(f"Best DE Solution: {chrom}")

    # Comparison
    with tabs[9]:
        st.header("GA vs DE Comparison")
        if st.session_state.ga_all_runs and st.session_state.de_all_runs:
            fig = go.Figure()
            for i, hist in enumerate(st.session_state.ga_all_runs):
                fig.add_trace(go.Scatter(
                    y=hist,
                    x=list(range(1, len(hist) + 1)),
                    mode='lines+markers',
                    name=f'GA Run {i + 1}',
                    hovertemplate='Gen %{x}<br>Fitness %{y:.4f}'
                ))
            for i, hist in enumerate(st.session_state.de_all_runs):
                fig.add_trace(go.Scatter(
                    y=hist,
                    x=list(range(1, len(hist) + 1)),
                    mode='lines+markers',
                    name=f'DE Run {i + 1}',
                    hovertemplate='Gen %{x}<br>Fitness %{y:.4f}'
                ))
            fig.update_layout(title="GA vs DE Comparison",
                              xaxis_title="Generation",
                              yaxis_title="Best Fitness",
                              width=900, height=500)
            st.plotly_chart(fig, use_container_width=True)

    # Visualization
    with tabs[10]:
        st.header("Resource Visualization")
        visualize_data()

    # Sidebar Random Data
    st.sidebar.header("Random Data Generation")
    ns = st.sidebar.number_input("Servers", min_value=1, value=5)
    nv = st.sidebar.number_input("VMs", min_value=1, value=10)
    nt = st.sidebar.number_input("Tasks", min_value=1, value=5)
    if st.sidebar.button("Generate Random Data"):
        st.session_state.servers, st.session_state.vms, st.session_state.tasks = generate_random_data(ns, nv, nt)
        st.success("Random data generated")

if __name__ == "__main__":
    main()
