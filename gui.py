# gui.py
import tkinter as tk
from collections import defaultdict
from copy import deepcopy
from tkinter import ttk, messagebox

import mplcursors
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import animation, pyplot as plt, cm

from de import DifferentialEvolution
from models import Server, VM, Task
from utils import generate_random_data
from simulation import run_ga_generator
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from tkinter import filedialog
import pandas as pd
import os
from tkinter import PhotoImage
from tkinter import PhotoImage
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloud Resource Allocation - GA")
        self.root.geometry("1200x700")
        self.servers = []
        self.vms = []
        self.tasks = []
        self._build_ui()
        self._runs_added = set()
        self.best_run_idx = 0
        # قوائم لتخزين جميع الـ Runs
        self.ga_all_runs = []  # كل Run من GA
        self.de_all_runs = []  # كل Run من DE

        # لمؤشرات أفضل Run
        self.best_ga_run_idx = None
        self.best_de_run_idx = None
        # default value
        self.run_colors = ["blue", "green", "red", "purple", "orange",
                           "brown", "cyan", "magenta", "gold", "darkgreen"]

        self.run_colors = {}

    def _build_ui(self):
        # Top: Welcome + mode buttons
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", pady=6)
        tk.Label(top, text="Cloud Resource Allocation — Genetic Algorithm", font=("Arial", 16, "bold")).pack(side="left", padx=8)
        btn_frame = tk.Frame(top)
        btn_frame.pack(side="right", padx=8)

        from tkinter import ttk

        # -----------------------------
        # تعريف ستايل للأزرار DE
        # -----------------------------
        style = ttk.Style()
        style.theme_use('default')  # يضمن توافق الألوان

        # ستايل زر أخضر للـ Load Random
        style.configure("Random.TButton",
                        foreground="white",  # لون النص
                        background="#4CAF50",  # لون الخلفية
                        font=("Arial", 10, "bold"),
                        padding=5)
        style.map("Random.TButton",
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '#45a049'), ('active', '#66BB6A')])

        # ستايل زر أزرق للـ Reset DE
        style.configure("Reset.TButton",
                        foreground="white",
                        background="#2196F3",
                        font=("Arial", 10, "bold"),
                        padding=5)
        style.map("Reset.TButton",
                  foreground=[('pressed', 'white'), ('active', 'white')],
                  background=[('pressed', '#1976D2'), ('active', '#64B5F6')])

        # -----------------------------
        # إضافة الأزرار إلى btn_frame
        # -----------------------------
        ttk.Button(btn_frame, text="Load Random", style="Random.TButton",
                   command=self.open_random_dialog).pack(side="right", padx=4)


        ttk.Button(btn_frame, text="Reset ALL", command=self.reset_all).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Reset GA & DE  ", command=self.reset_ui_only).pack(side="right", padx=4)

        # Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)
        style = ttk.Style()
        style.theme_create("my_custom_notebook", parent="alt", settings={
            "TNotebook": {
                "configure": {
                    "tabmargins": [2, 5, 2, 0],
                }
            },
            "TNotebook.Tab": {
                "configure": {
                    "padding": [20, 10],
                    "background": "#3a3a3a",  # لون التاب العادي
                    "foreground": "white"
                },
                "map": {
                    "background": [("selected", "#0078ff")],  # لون التاب المختار
                    "foreground": [("selected", "white")],
                }
            }
        })

        style.theme_use("my_custom_notebook")

        # Tabs
        self.tab_servers = tk.Frame(self.nb)
        self.tab_vms = tk.Frame(self.nb)
        self.tab_tasks = tk.Frame(self.nb)
        self.tab_ga = tk.Frame(self.nb)
        self.tab_run = tk.Frame(self.nb)

        self.nb.add(self.tab_servers, text="Servers")
        self.nb.add(self.tab_vms, text="VMs")
        self.nb.add(self.tab_tasks, text="Tasks")
        self.nb.add(self.tab_ga, text="GA Parameters")
        self.nb.add(self.tab_run, text="GA Run & output")
        self.tab_de_params = tk.Frame(self.nb)
        self.tab_de_results = tk.Frame(self.nb)
        self.tab_de_params = tk.Frame(self.nb)
        self.tab_de_results = tk.Frame(self.nb)

        self._build_de_params_tab()
        self._build_de_results_tab()


        self._build_servers_tab()
        self._build_vms_tab()
        self._build_tasks_tab()
        self._build_ga_tab()
        self._build_run_tab()
        # ---------------- Summary tab ----------------
        self.tab_summary = tk.Frame(self.nb)
        self.nb.add(self.tab_summary, text="GA Summary")
        self.summary_frame = tk.Frame(self.tab_summary)
        self.summary_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.nb.add(self.tab_de_params, text="DE Parameters")
        self.nb.add(self.tab_de_results, text="DE Run & output")
        self.tab_de_mapping = ttk.Frame(self.nb)
        self.nb.add(self.tab_de_mapping, text="DE Summary")

        self.tab_comparison = tk.Frame(self.nb)
        self.nb.add(self.tab_comparison, text="GA vs DE Comparison")
        self.build_compare_tab()

    def reset_all(self):
        # ----------------------
        # Reset main data
        # ----------------------
        self.servers = []
        self.vms = []
        self.tasks = []
        """
                Reset GA-related GUI elements without touching DE elements or main data
                """
        # ----------------------
        # Clear Treeviews for GA only
        # ----------------------
        ga_treeviews = ('tv_servers', 'tv_vms', 'tv_tasks')
        for tv_name in ga_treeviews:
            if hasattr(self, tv_name):
                tv = getattr(self, tv_name)
                for row in tv.get_children():
                    tv.delete(row)

        # ----------------------
        # Clear GA output areas / plots if any
        # ----------------------
        self._clear_output_area()
        self.update_server_auto_id()

        # ----------------------
        # Update GA-related visuals / graphs
        # ----------------------
        self.update_env_visualization()
        self._update_server_graph()
        self._update_task_graph()

        # Reset GA timer label
        if hasattr(self, 'lbl_ga_time'):
            self.lbl_ga_time.config(text="0.0 s")
        """Reset DE-related GUI elements without touching GA elements or main data"""
        self.de_running = False

        # Reset timer
        if hasattr(self, 'lbl_de_time'):
            self.lbl_de_time.config(text="0.0 s")

        # Clear DE Treeviews
        for tv in (self.tv_task_vm, self.tv_history):
            if hasattr(self, tv):
                tree = getattr(self, tv)
                for row in tree.get_children():
                    tree.delete(row)

        # Clear DE plots
        if hasattr(self, 'de_canvas_frame'):
            for widget in self.de_canvas_frame.winfo_children():
                widget.destroy()

        # Clear VM → Task map frames
        for f in ('chrom_frame', 'vm_map_frame', 'tab_de_mapping'):
            if hasattr(self, f):
                frame = getattr(self, f)
                for widget in frame.winfo_children():
                    widget.destroy()



    def reset_ui_only(self):
        """Reset DE-related GUI elements without touching GA elements or main data"""
        self.de_running = False

        # Reset timer
        if hasattr(self, 'lbl_de_time'):
            self.lbl_de_time.config(text="0.0 s")

        # Clear DE Treeviews
        for tv in (self.tv_task_vm, self.tv_history):
            if hasattr(self, tv):
                tree = getattr(self, tv)
                for row in tree.get_children():
                    tree.delete(row)

        # Clear DE plots
        if hasattr(self, 'de_canvas_frame'):
            for widget in self.de_canvas_frame.winfo_children():
                widget.destroy()

        # Clear VM → Task map frames
        for f in ('chrom_frame', 'vm_map_frame', 'tab_de_mapping'):
            if hasattr(self, f):
                frame = getattr(self, f)
                for widget in frame.winfo_children():
                    widget.destroy()
        """
                Reset GA-related GUI elements without touching DE elements or main data
                """
        # ----------------------
        # Clear Treeviews for GA only
        # ----------------------
        ga_treeviews = ('tv_servers', 'tv_vms', 'tv_tasks')
        for tv_name in ga_treeviews:
            if hasattr(self, tv_name):
                tv = getattr(self, tv_name)
                for row in tv.get_children():
                    tv.delete(row)

        # ----------------------
        # Clear GA output areas / plots if any
        # ----------------------
        self._clear_output_area()
        self.update_server_auto_id()

        # ----------------------
        # Update GA-related visuals / graphs
        # ----------------------
        self.update_env_visualization()
        self._update_server_graph()
        self._update_task_graph()

        # Reset GA timer label
        if hasattr(self, 'lbl_ga_time'):
            self.lbl_ga_time.config(text="0.0 s")

    def reset_ui_GA(self):
        """
        Reset GA-related GUI elements without touching DE elements or main data
        """
        # ----------------------
        # Clear Treeviews for GA only
        # ----------------------
        ga_treeviews = ('tv_servers', 'tv_vms', 'tv_tasks')
        for tv_name in ga_treeviews:
            if hasattr(self, tv_name):
                tv = getattr(self, tv_name)
                for row in tv.get_children():
                    tv.delete(row)

        # ----------------------
        # Clear GA output areas / plots if any
        # ----------------------
        self._clear_output_area()
        self.update_server_auto_id()



        # ----------------------
        # Update GA-related visuals / graphs
        # ----------------------
        self.update_env_visualization()
        self._update_server_graph()
        self._update_task_graph()

        # Reset GA timer label
        if hasattr(self, 'lbl_ga_time'):
            self.lbl_ga_time.config(text="0.0 s")


    def reset_ui_DE(self):
        """Reset DE-related GUI elements without touching GA elements or main data"""
        self.de_running = False

        # Reset timer
        if hasattr(self, 'lbl_de_time'):
            self.lbl_de_time.config(text="0.0 s")

        # Clear DE Treeviews
        for tv in (self.tv_task_vm, self.tv_history):
            if hasattr(self, tv):
                tree = getattr(self, tv)
                for row in tree.get_children():
                    tree.delete(row)

        # Clear DE plots
        if hasattr(self, 'de_canvas_frame'):
            for widget in self.de_canvas_frame.winfo_children():
                widget.destroy()

        # Clear VM → Task map frames
        for f in ('chrom_frame', 'vm_map_frame', 'tab_de_mapping'):
            if hasattr(self, f):
                frame = getattr(self, f)
                for widget in frame.winfo_children():
                    widget.destroy()

    # # ---------------- Servers tab ----------------
    def _build_servers_tab(self):
        frm = tk.Frame(self.tab_servers)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        left = tk.Frame(frm)
        left.pack(side="left", fill="y", padx=6)

        tk.Label(left, text="Add Server", font=("Arial", 11, "bold")).pack(anchor="w")
        f = tk.Frame(left); f.pack(anchor="w", pady=4)
        tk.Label(f, text="ID").grid(row=0, column=0)
        tk.Label(f, text="CPU").grid(row=0, column=1)
        tk.Label(f, text="RAM").grid(row=0, column=2)
        tk.Label(f, text="Storage").grid(row=0, column=3)
        tk.Label(f, text="Cost").grid(row=0, column=4)

        self.e_srv_id = tk.Entry(f, width=6); self.e_srv_id.grid(row=1, column=0, padx=4)
        self.e_srv_cpu = tk.Entry(f, width=6); self.e_srv_cpu.grid(row=1, column=1, padx=4)
        self.e_srv_ram = tk.Entry(f, width=6); self.e_srv_ram.grid(row=1, column=2, padx=4)
        self.e_srv_st = tk.Entry(f, width=8); self.e_srv_st.grid(row=1, column=3, padx=4)
        self.e_srv_cost = tk.Entry(f, width=8); self.e_srv_cost.grid(row=1, column=4, padx=4)



        ttk.Button(left, text="Add Server", command=self.add_server).pack(pady=6)
        ttk.Button(left, text="Add Server", command=self.add_server).pack(pady=6)

        right = tk.Frame(frm)
        right.pack(side="left", fill="both", expand=True, padx=6)
        cols = ("id", "cpu", "ram", "storage", "cost", "remaining", "rem_cost")
        self.tv_servers = ttk.Treeview(right, columns=cols, show="headings", height=10)

        for c in cols:
            self.tv_servers.heading(c, text=c.title())
            self.tv_servers.column(c, width=110)
        self.tv_servers.pack(fill="both", expand=True)


    def update_server_auto_id(self):
        # احسب أعلى ID موجود بين السيرفرات
        next_id = max([s.id for s in self.servers], default=-1) + 1
        # ضع الرقم في الـ Entry
        self.e_srv_id.delete(0, tk.END)
        self.e_srv_id.insert(0, str(next_id))

    def add_server(self):
        try:
            # ID تلقائي لو فارغ
            id_text = self.e_srv_id.get().strip()
            if id_text == "":
                sid = max([s.id for s in self.servers], default=-1) + 1
            else:
                sid = int(id_text)

            # بقية القيم مع إزالة المسافات
            cpu_text = self.e_srv_cpu.get().strip()
            ram_text = self.e_srv_ram.get().strip()
            st_text = self.e_srv_st.get().strip()
            cost_text = self.e_srv_cost.get().strip()

            # تحقق من أن كل القيم ليست فارغة
            if not cpu_text or not ram_text or not st_text or not cost_text:
                raise ValueError("Empty field")

            cpu = int(cpu_text)
            ram = int(ram_text)
            st = int(st_text)
            cost = float(cost_text)

        except ValueError as e:
            print(f"Debug: Exception: {e}")  # طباعة السبب في الكونسول
            messagebox.showerror("Invalid input", "Enter valid numeric server values")
            return

        # التحقق من ID مكرر
        if any(s.id == sid for s in self.servers):
            messagebox.showerror("Duplicate ID", "Server ID already exists")
            return

        # إضافة السيرفر
        s = Server(sid, cpu, ram, st, cost)
        self.servers.append(s)
        self._update_server_tree()
        self._refresh_vm_server_choices()
        self.update_server_auto_id()

        self.e_srv_cpu.delete(0, tk.END)
        self.e_srv_ram.delete(0, tk.END)
        self.e_srv_st.delete(0, tk.END)
        self.e_srv_cost.delete(0, tk.END)

    def _update_server_tree(self):
        # Clear existing rows
        for i in self.tv_servers.get_children():
            self.tv_servers.delete(i)

        for s in self.servers:
            rem = s.remaining()  # CPU, RAM, Storage remaining
            remaining_cost = s.cost - sum(vm.cost for vm in s.vms)
            if remaining_cost < 0:
                remaining_cost = 0

            # Insert row with old remaining column + new rem_cost column
            self.tv_servers.insert("", "end", values=(
                s.id,
                s.cpu,
                s.ram,
                s.storage,
                s.cost,
                f"CPU:{rem['cpu']}   /  RAM:{rem['ram']}   /  ST:{rem['storage']}",  # old remaining info
                round(remaining_cost, 3)  # new remaining cost
            ))
    def _build_servers_tab(self):
        frm = tk.Frame(self.tab_servers)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        # ---------------- Left panel ----------------
        left = tk.Frame(frm)
        left.pack(side="left", fill="y", padx=6)

        tk.Label(left, text="Add Server", font=("Arial", 11, "bold")).pack(anchor="w")
        f = tk.Frame(left)
        f.pack(anchor="w", pady=4)

        tk.Label(f, text="ID").grid(row=0, column=0)
        tk.Label(f, text="CPU").grid(row=0, column=1)
        tk.Label(f, text="RAM").grid(row=0, column=2)
        tk.Label(f, text="Storage").grid(row=0, column=3)
        tk.Label(f, text="Cost").grid(row=0, column=4)

        self.e_srv_id = tk.Entry(f, width=6)
        self.e_srv_id.grid(row=1, column=0, padx=4)
        self.e_srv_cpu = tk.Entry(f, width=6)
        self.e_srv_cpu.grid(row=1, column=1, padx=4)
        self.e_srv_ram = tk.Entry(f, width=6)
        self.e_srv_ram.grid(row=1, column=2, padx=4)
        self.e_srv_st = tk.Entry(f, width=8)
        self.e_srv_st.grid(row=1, column=3, padx=4)
        self.e_srv_cost = tk.Entry(f, width=8)
        self.e_srv_cost.grid(row=1, column=4, padx=4)

        ttk.Button(left, text="Add Server", command=self.add_server).pack(pady=6)

        # ---------------- Right panel ----------------
        right = tk.Frame(frm)
        right.pack(side="left", fill="both", expand=True, padx=6)

        # Treeview for servers
        cols = ("id", "cpu", "ram", "storage", "cost", "remaining", "rem_cost")
        self.tv_servers = ttk.Treeview(right, columns=cols, show="headings", height=10)

        for c in cols:
            self.tv_servers.heading(c, text=c.title())
            self.tv_servers.column(c, width=110)

        self.tv_servers.pack(fill="both", expand=True)

        # ---------------- Server Graph ----------------
        tk.Label(right, text="Server Graph", font=("Arial", 11, "bold")).pack(pady=(10, 0))

        canvas_frame = tk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        self.server_canvas = tk.Canvas(canvas_frame, bg="#f0f0f0", height=200)
        self.server_canvas.pack(side="left", fill="both", expand=True)

        self.scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.server_canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.server_canvas.configure(yscrollcommand=self.scroll_y.set)

        self.graph_frame = tk.Frame(self.server_canvas, bg="#f0f0f0")
        self.server_canvas.create_window((0, 0), window=self.graph_frame, anchor="nw")

        # Frame للتعامل مع scroll
        canvas_frame = tk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        # Initial draw
        self._update_server_graph()


    def _update_server_graph(self):
        import os
        import tkinter as tk
        from tkinter import PhotoImage

        # Clear previous graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Parameters
        card_w, card_h = 200,200
        padding = 10

        # Load icons
        BASE_DIR = os.path.dirname(__file__)
        cpu_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "chip.png"))
        ram_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "ram.png"))
        storage_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "database.png"))
        time_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "clock.png"))
        cost_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "save-money.png"))

        # Canvas for horizontal scrolling
        canvas = tk.Canvas(self.graph_frame, bg="#f0f0f0", height=card_h + 2 * padding , width=1025, highlightthickness=0)
        canvas.pack(side="top", fill="both", expand=True)

        h_scroll = tk.Scrollbar(self.graph_frame, orient="horizontal", command=canvas.xview)
        h_scroll.pack(side="bottom", fill="x")
        canvas.configure(xscrollcommand=h_scroll.set)

        inner_frame = tk.Frame(canvas, bg="#f0f0f0")
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Draw servers
        for server in self.servers:

            frame = tk.LabelFrame(inner_frame, text=f"Server {server.id}",
                                  bg="#ffffff", padx=10, pady=10, font=("Arial", 10, "bold"),
                                  width=card_w, height=card_h)
            frame.pack_propagate(False)
            frame.pack(side="left", padx=padding, pady=padding)

            # CPU
            cpu_frame = tk.Frame(frame, bg="#ffffff")
            cpu_frame.pack(anchor="w")
            tk.Label(cpu_frame, image=cpu_icon, bg="#ffffff").pack(side="left")
            tk.Label(cpu_frame, text=f"CPU: {server.cpu}", bg="#ffffff").pack(side="left", padx=5)

            # RAM
            ram_frame = tk.Frame(frame, bg="#ffffff")
            ram_frame.pack(anchor="w")
            tk.Label(ram_frame, image=ram_icon, bg="#ffffff").pack(side="left")
            tk.Label(ram_frame, text=f"RAM: {server.ram}", bg="#ffffff").pack(side="left", padx=5)

            # Storage
            storage_frame = tk.Frame(frame, bg="#ffffff")
            storage_frame.pack(anchor="w")
            tk.Label(storage_frame, image=storage_icon, bg="#ffffff").pack(side="left")
            tk.Label(storage_frame, text=f"Storage: {server.storage}", bg="#ffffff").pack(side="left", padx=5)

            # Cost
            cost_frame = tk.Frame(frame, bg="#ffffff")
            cost_frame.pack(anchor="w")
            tk.Label(cost_frame, image=cost_icon, bg="#ffffff").pack(side="left")
            tk.Label(cost_frame, text=f"Cost: {server.cost}", bg="#ffffff").pack(side="left", padx=5)

        # Keep references to icons to prevent garbage collection
        self._server_icons = [cpu_icon, ram_icon, storage_icon, time_icon, cost_icon]

        # Update scrollregion
        inner_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))


    def _update_server_tree(self):
        # Clear existing rows
        for i in self.tv_servers.get_children():
            self.tv_servers.delete(i)

        for s in self.servers:
            rem = s.remaining()  # CPU, RAM, Storage remaining
            remaining_cost = s.cost - sum(vm.cost for vm in s.vms)
            if remaining_cost < 0:
                remaining_cost = 0

            # Insert row with old remaining column + new rem_cost column
            self.tv_servers.insert("", "end", values=(
                s.id,
                s.cpu,
                s.ram,
                s.storage,
                s.cost,
                f"CPU:{rem['cpu']}   /  RAM:{rem['ram']}   /  ST:{rem['storage']}",  # old remaining info
                round(remaining_cost, 3)  # new remaining cost
            ))
        self.update_env_visualization()
        self._update_server_graph()
        self._update_task_graph()
        self.update_server_remaining()

    # ---------------- VMs tab ----------------
    def _build_vms_tab(self):
        frm = tk.Frame(self.tab_vms)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        # ---------------- Left panel ----------------
        left = tk.Frame(frm)
        left.pack(side="left", fill="y", padx=6)

        # Add VM section
        tk.Label(left, text="Add VM", font=("Arial", 11, "bold")).pack(anchor="w")
        f = tk.Frame(left)
        f.pack(anchor="w", pady=4)
        tk.Label(f, text="ID").grid(row=0, column=0)
        tk.Label(f, text="Server").grid(row=0, column=1)
        tk.Label(f, text="CPU").grid(row=0, column=2)
        tk.Label(f, text="RAM").grid(row=0, column=3)
        tk.Label(f, text="Storage").grid(row=0, column=4)
        tk.Label(f, text="Cost").grid(row=0, column=5)

        self.e_vm_id = tk.Entry(f, width=6)
        self.e_vm_id.grid(row=1, column=0, padx=4)
        self.cb_vm_server = ttk.Combobox(f, values=[], width=8, state="readonly")
        self.cb_vm_server.grid(row=1, column=1, padx=4)
        self.e_vm_cpu = tk.Entry(f, width=6)
        self.e_vm_cpu.grid(row=1, column=2, padx=4)
        self.e_vm_ram = tk.Entry(f, width=6)
        self.e_vm_ram.grid(row=1, column=3, padx=4)
        self.e_vm_st = tk.Entry(f, width=8)
        self.e_vm_st.grid(row=1, column=4, padx=4)
        self.e_vm_cost = tk.Entry(f, width=8)
        self.e_vm_cost.grid(row=1, column=5, padx=4)

        ttk.Button(left, text="Add VM", command=self.add_vm).pack(pady=6)

        # Server remaining resources table
        tk.Label(left, text="Server Remaining Resources").pack(pady=(10, 0))
        self.tv_server_remaining = ttk.Treeview(left, columns=("CPU", "RAM", "Storage", "Cost"),
                                                show="headings", height=1)
        for c in ("CPU", "RAM", "Storage", "Cost"):
            self.tv_server_remaining.heading(c, text=c)
            self.tv_server_remaining.column(c, width=80)
        self.tv_server_remaining.pack(pady=4)
        self.update_vm_auto_id()

        self.cb_vm_server.bind("<<ComboboxSelected>>", self.update_server_remaining)

        # ---------------- Right panel ----------------
        right = tk.Frame(frm)
        right.pack(side="left", fill="both", expand=True, padx=6)

        # Treeview for VMs
        cols = ("id", "server", "cpu", "ram", "storage", "cost")
        self.tv_vms = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c in cols:
            self.tv_vms.heading(c, text=c.title())
            self.tv_vms.column(c, width=110)
        self.tv_vms.pack(fill="x", pady=4)

        # Label for environment visualization
        tk.Label(right, text="Server → VM View", font=("Arial", 11, "bold")).pack(pady=(10, 0))

        # ----------- Scrollable canvas (FULL EXPANDED AREA) -----------
        canvas_frame = tk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        # Canvas fills everything
        self.env_canvas = tk.Canvas(canvas_frame, bg="#f0f0f0")
        self.env_canvas.pack(side="left", fill="both", expand=True)

        # Vertical Scrollbar
        self.env_scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.env_canvas.yview)
        self.env_scroll_y.pack(side="right", fill="y")

        # Horizontal Scrollbar
        self.env_scroll_x = tk.Scrollbar(right, orient="horizontal", command=self.env_canvas.xview)
        self.env_scroll_x.pack(side="bottom", fill="x")

        self.env_canvas.configure(yscrollcommand=self.env_scroll_y.set,
                                  xscrollcommand=self.env_scroll_x.set)

        # Frame inside canvas
        self.env_frame = tk.Frame(self.env_canvas, bg="#f0f0f0")
        self.env_canvas.create_window((0, 0), window=self.env_frame, anchor="nw")

        # Update scrollregion dynamically
        self.env_frame.bind("<Configure>",
                            lambda e: self.env_canvas.configure(scrollregion=self.env_canvas.bbox("all")))

        # Show environment immediately
        self.update_env_visualization()

    # ---------------- Helper Methods ----------------
    def _refresh_vm_server_choices(self):
        vals = [str(s.id) for s in self.servers]
        self.cb_vm_server['values'] = vals

    def update_server_remaining(self, event=None):
        self.tv_server_remaining.delete(*self.tv_server_remaining.get_children())

        if not self.cb_vm_server.get():
            return

        try:
            server_id = int(self.cb_vm_server.get())
        except ValueError:
            return

        server = next((s for s in self.servers if s.id == server_id), None)
        if server:
            rem = server.remaining()  # CPU, RAM, Storage remaining
            remaining_cost = server.cost - sum(vm.cost for vm in server.vms)
            if remaining_cost < 0:
                remaining_cost = 0

            self.tv_server_remaining.insert("", "end",
                                            values=(rem["cpu"], rem["ram"], rem["storage"], remaining_cost))

    def add_vm(self):
        try:
            # --- ID تلقائي ---

            id_text = self.e_vm_id.get().strip()
            if id_text == "":
                vid = max([v.id for v in self.vms], default=-1) + 1
            else:
                vid = int(id_text)

            # بقية القيم
            server_id = int(self.cb_vm_server.get())
            cpu = int(self.e_vm_cpu.get().strip())
            ram = int(self.e_vm_ram.get().strip())
            st = int(self.e_vm_st.get().strip())
            cost = float(self.e_vm_cost.get().strip())

        except:
            messagebox.showerror("Invalid input", "Enter valid VM values and select server")
            return

        # العثور على السيرفر
        server = next((s for s in self.servers if s.id == server_id), None)
        if server is None:
            messagebox.showerror("Server not found", "Select a valid server")
            return

        # منع تكرار ID
        if any(v.id == vid for v in self.vms):
            messagebox.showerror("Duplicate ID", "VM ID already exists")
            return

        # التحقق من قدرة السيرفر على استضافة VM
        if not server.can_add_vm(cpu, ram, st, cost):
            msg = (
                f"Server {server_id} cannot host this VM due to:\n"
                f"- Insufficient CPU/RAM/Storage OR\n"
                f"- Cost limit exceeded\n\n"
                f"Server max cost: {server.cost}\n"
                f"Current VMs cost: {server.total_vm_cost()}\n"
                f"Requested VM cost: {cost}\n"
            )
            messagebox.showerror("Capacity exceeded", msg)
            return

        # إنشاء الـ VM وإضافته للسيرفر
        vm = VM(vid, server_id, cpu, ram, st, cost)
        self.vms.append(vm)
        server.vms.append(vm)

        # تحديث واجهة المستخدم
        self.tv_vms.insert("", "end",
                           values=(vm.id, vm.server_id, vm.cpu, vm.ram, vm.storage, vm.cost))
        self.update_env_visualization()
        self.update_server_remaining()
        self.update_env_visualization()
        self.update_server_remaining()
        self._update_server_tree()
        self.update_vm_auto_id()

        self.e_vm_cpu.delete(0, tk.END)
        self.e_vm_ram.delete(0, tk.END)
        self.e_vm_st.delete(0, tk.END)
        self.e_vm_cost.delete(0, tk.END)

    def update_vm_auto_id(self):
        # احسب أعلى ID موجود
        next_id = max([v.id for v in self.vms], default=-1) + 1
        # ضع الرقم في الـ Entry
        self.e_vm_id.delete(0, tk.END)
        self.e_vm_id.insert(0, str(next_id))


    def update_env_visualization(self):
        import os
        import tkinter as tk
        from tkinter import PhotoImage

        # Clear previous visualization
        for widget in self.env_frame.winfo_children():
            widget.destroy()

        # Load icons
        BASE_DIR = os.path.dirname(__file__)
        cpu_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "chip.png"))
        ram_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "ram.png"))
        storage_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "database.png"))
        time_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "clock.png"))
        cost_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "save-money.png"))

        for server in self.servers:
            # Server frame
            server_frame = tk.LabelFrame(self.env_frame, text=f"Server {server.id}",
                                         padx=10, pady=10, bg="#d0e1f9", font=("Arial", 10, "bold"))
            server_frame.pack(side="left", padx=10, pady=10, fill="y")

            if not server.vms:
                tk.Label(server_frame, text="No VMs", bg="#d0e1f9").pack(padx=5, pady=5)
            else:
                for vm in server.vms:
                    vm_frame = tk.LabelFrame(server_frame, text=f"VM {vm.id}", padx=5, pady=5, bg="#ffffff")
                    vm_frame.pack(pady=5, fill="both")

                    # CPU
                    cpu_frame = tk.Frame(vm_frame, bg="#ffffff")
                    cpu_frame.pack(anchor="w", pady=2)
                    tk.Label(cpu_frame, image=cpu_icon, bg="#ffffff").pack(side="left")
                    tk.Label(cpu_frame, text=f"CPU: {vm.cpu}", bg="#ffffff").pack(side="left", padx=5)

                    # RAM
                    ram_frame = tk.Frame(vm_frame, bg="#ffffff")
                    ram_frame.pack(anchor="w", pady=2)
                    tk.Label(ram_frame, image=ram_icon, bg="#ffffff").pack(side="left")
                    tk.Label(ram_frame, text=f"RAM: {vm.ram}", bg="#ffffff").pack(side="left", padx=5)

                    # Storage
                    storage_frame = tk.Frame(vm_frame, bg="#ffffff")
                    storage_frame.pack(anchor="w", pady=2)
                    tk.Label(storage_frame, image=storage_icon, bg="#ffffff").pack(side="left")
                    tk.Label(storage_frame, text=f"Storage: {vm.storage}", bg="#ffffff").pack(side="left", padx=5)

                    # Cost
                    cost_frame = tk.Frame(vm_frame, bg="#ffffff")
                    cost_frame.pack(anchor="w", pady=2)
                    tk.Label(cost_frame, image=cost_icon, bg="#ffffff").pack(side="left")
                    tk.Label(cost_frame, text=f"Cost: {vm.cost}", bg="#ffffff").pack(side="left", padx=5)

                    # Time (if you have a time attribute)
                    if hasattr(vm, "time"):
                        time_frame = tk.Frame(vm_frame, bg="#ffffff")
                        time_frame.pack(anchor="w", pady=2)
                        tk.Label(time_frame, image=time_icon, bg="#ffffff").pack(side="left")
                        tk.Label(time_frame, text=f"Time: {vm.time}", bg="#ffffff").pack(side="left", padx=5)

        # Keep references to icons to prevent garbage collection
        self._env_icons = [cpu_icon, ram_icon, storage_icon, time_icon, cost_icon]

    # ---------------- Tasks tab ----------------
    def _build_tasks_tab(self):
        frm = tk.Frame(self.tab_tasks)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        # ---------------- Left panel ----------------
        left = tk.Frame(frm)
        left.pack(side="left", fill="y", padx=6)

        tk.Label(left, text="Add Task", font=("Arial", 11, "bold")).pack(anchor="w")
        f = tk.Frame(left)
        f.pack(anchor="w", pady=4)

        tk.Label(f, text="ID").grid(row=0, column=0)
        tk.Label(f, text="CPU").grid(row=0, column=1)
        tk.Label(f, text="RAM").grid(row=0, column=2)
        tk.Label(f, text="Storage").grid(row=0, column=3)
        tk.Label(f, text="Time").grid(row=0, column=4)
        tk.Label(f, text="Cost").grid(row=0, column=5)
        tk.Button(frm, text="Load Dataset",
                  bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold"),
                  command=self.load_dataset).pack(side="left", padx=5)

        self.e_task_id = tk.Entry(f, width=6);
        self.e_task_id.grid(row=1, column=0, padx=4)
        self.e_task_cpu = tk.Entry(f, width=6);
        self.e_task_cpu.grid(row=1, column=1, padx=4)
        self.e_task_ram = tk.Entry(f, width=6);
        self.e_task_ram.grid(row=1, column=2, padx=4)
        self.e_task_st = tk.Entry(f, width=8);
        self.e_task_st.grid(row=1, column=3, padx=4)
        self.e_task_time = tk.Entry(f, width=6);
        self.e_task_time.grid(row=1, column=4, padx=4)
        self.e_task_cost = tk.Entry(f, width=8);
        self.e_task_cost.grid(row=1, column=5, padx=4)
        self.update_task_auto_id()
        ttk.Button(left, text="Add Task", command=self.add_task).pack(pady=6)

        # ---------------- Right panel ----------------
        right = tk.Frame(frm)
        right.pack(side="left", fill="both", expand=True, padx=6)

        # Treeview for tasks
        cols = ("id", "cpu", "ram", "storage", "time", "cost")
        self.tv_tasks = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c in cols:
            self.tv_tasks.heading(c, text=c.title())
            self.tv_tasks.column(c, width=110)
        self.tv_tasks.pack(fill="both", expand=True)

        # ---------------- Tasks Graph ----------------
        tk.Label(right, text="Tasks Graph", font=("Arial", 11, "bold")).pack(pady=(10, 0))

        canvas_frame = tk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)

        self.task_canvas = tk.Canvas(canvas_frame, bg="#f0f0f0", height=200)
        self.task_canvas.pack(side="left", fill="both", expand=True)

        self.task_scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.task_canvas.yview)
        self.task_scroll_y.pack(side="right", fill="y")
        self.task_canvas.configure(yscrollcommand=self.task_scroll_y.set)

        self.task_graph_frame = tk.Frame(self.task_canvas, bg="#f0f0f0")
        self.task_canvas.create_window((0, 0), window=self.task_graph_frame, anchor="nw")

        self.task_graph_frame.bind("<Configure>",
                                   lambda e: self.task_canvas.configure(scrollregion=self.task_canvas.bbox("all")))


        # Initial draw
        self._update_task_graph()
        self.update_task_auto_id()

    def add_task(self):
        try:
            # --- ID تلقائي ---
            id_text = self.e_task_id.get().strip()
            if id_text == "":
                tid = max([t.id for t in self.tasks], default=-1) + 1
                self.e_task_id.delete(0, tk.END)
                self.e_task_id.insert(0, str(tid))  # عرض الرقم في الـ Entry
            else:
                tid = int(id_text)

            # بقية القيم
            cpu = int(self.e_task_cpu.get().strip())
            ram = int(self.e_task_ram.get().strip())
            st = int(self.e_task_st.get().strip())
            time_t = float(self.e_task_time.get().strip())
            cost = float(self.e_task_cost.get().strip())

        except Exception:
            messagebox.showerror("Invalid input", "Enter valid Task values")
            return

        # منع تكرار ID
        if any(t.id == tid for t in self.tasks):
            messagebox.showerror("Duplicate ID", "Task ID already exists")
            return

        # إضافة الـ Task
        task = Task(tid, cpu, ram, st, time_t, cost)
        self.tasks.append(task)

        # تحديث واجهة المستخدم
        self.tv_tasks.insert("", "end", values=(task.id, task.cpu, task.ram, task.storage, task.time, task.cost))
        self._update_task_graph()

        # مسح الحقول بعد الإضافة
        self.e_task_cpu.delete(0, tk.END)
        self.e_task_ram.delete(0, tk.END)
        self.e_task_st.delete(0, tk.END)
        self.e_task_time.delete(0, tk.END)
        self.e_task_cost.delete(0, tk.END)

        # وضع ID تلقائي جديد بعد المسح
        self.update_task_auto_id()

    def update_task_auto_id(self):
        next_id = max([t.id for t in self.tasks], default=-1) + 1
        self.e_task_id.delete(0, tk.END)
        self.e_task_id.insert(0, str(next_id))

    def _update_task_graph(self):
        import os
        from tkinter import PhotoImage, Canvas, Frame, ttk

        def create_rounded_rect(canvas, x1, y1, x2, y2, r=10, **kwargs):
            points = [
                x1 + r, y1,
                x2 - r, y1,
                x2, y1,
                x2, y1 + r,
                x2, y2 - r,
                x2, y2,
                x2 - r, y2,
                x1 + r, y2,
                x1, y2,
                x1, y2 - r,
                x1, y1 + r,
                x1, y1
            ]
            return canvas.create_polygon(points, smooth=True, **kwargs)

        BASE_DIR = os.path.dirname(__file__)
        cpu_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "chip.png"))
        ram_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "ram.png"))
        storage_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "database.png"))
        time_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "clock.png"))
        cost_icon = PhotoImage(file=os.path.join(BASE_DIR, "icon", "save-money.png"))

        for widget in self.task_graph_frame.winfo_children():
            widget.destroy()

        canvas = Canvas(self.task_graph_frame, bg="#f0f0f0",width=880,  highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(self.task_graph_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=h_scrollbar.set)
        canvas.pack(side="top", fill="both", expand=True)
        h_scrollbar.pack(side="bottom", fill="x")

        scrollable_frame = Frame(canvas, bg="#f0f0f0")
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # ---------------- Card settings (صغيرة) ----------------
        card_width = 180  # أقل من السابق
        card_height = 220  # أقل من السابق
        padding = 10
        spacing = 35

        for task in self.tasks:
            frame = Frame(scrollable_frame, width=card_width, height=card_height, bg="#ffffff")
            frame.pack(side="left", padx=padding, pady=padding)
            frame.pack_propagate(False)

            c = Canvas(frame, width=card_width, height=card_height, bg="#ffffff", highlightthickness=0)
            c.pack(fill="both", expand=True)

            create_rounded_rect(c, 0, 0, card_width, card_height, r=15, fill="#ffffff", outline="#ccc", width=2)
            c.create_text(card_width // 2, 20, text=f"Task {task.id}", font=("Arial", 12, "bold"), fill="#00796b")

            start_y = 50
            items = [
                (cpu_icon, f"CPU: {task.cpu}"),
                (ram_icon, f"RAM: {task.ram}"),
                (storage_icon, f"Storage: {task.storage}"),
                (time_icon, f"Time: {task.time} sec"),
                (cost_icon, f"Cost: ${task.cost}")
            ]

            for icon, text in items:
                c.create_image(20, start_y, image=icon, anchor="w")
                c.create_text(55, start_y, text=text, anchor="w", font=("Arial", 9, "bold"))
                start_y += spacing

        self.task_icons = [cpu_icon, ram_icon, storage_icon, time_icon, cost_icon]

    # ---------------- GA tab ----------------
    def _build_ga_tab(self):
        frm = tk.Frame(self.tab_ga); frm.pack(padx=8, pady=8, fill="both", expand=True)
        left = tk.Frame(frm); left.pack(side="left", anchor="n", padx=6)
        tk.Label(left, text="GA Parameters", font=("Arial", 12, "bold")).pack(anchor="w")
        f = tk.Frame(left); f.pack(anchor="w", pady=6)
        tk.Label(f, text="Population").grid(row=0, column=0)
        tk.Label(f, text="Generations").grid(row=1, column=0)
        tk.Label(f, text="Crossover points").grid(row=2, column=0)
        tk.Label(f, text="Mutation rate (0-1)").grid(row=3, column=0)
        tk.Label(f, text="Elitism (keep top k)").grid(row=4, column=0)
        tk.Label(f, text="Time limit (sec, optional)").grid(row=5, column=0)

        self.e_pop = tk.Entry(f, width=8); self.e_pop.grid(row=0, column=1, padx=6)
        self.e_gens = tk.Entry(f, width=8); self.e_gens.grid(row=1, column=1, padx=6)
        self.e_cpoints = tk.Entry(f, width=8); self.e_cpoints.grid(row=2, column=1, padx=6)
        self.e_mrate = tk.Entry(f, width=8); self.e_mrate.grid(row=3, column=1, padx=6)
        self.e_elit = tk.Entry(f, width=8); self.e_elit.grid(row=4, column=1, padx=6)
        tk.Label(f, text="Number of Runs").grid(row=6, column=0)
        self.e_runs = tk.Entry(f, width=8)
        self.e_runs.grid(row=6, column=1, padx=6)
        self.e_runs.insert(0, "1")
        self.e_time = tk.Entry(f, width=8); self.e_time.grid(row=5, column=1, padx=6)

        self.e_pop.insert(0, "100")
        self.e_gens.insert(0, "100")
        self.e_cpoints.insert(0, "1")
        self.e_mrate.insert(0, "0.05")
        self.e_elit.insert(0, "1")
        self.e_time.insert(0, "")


        ttk.Button(left, text="Run GA (in Run tab)", command=lambda: self.nb.select(self.tab_run)).pack(pady=8)


    def _clear_output_area(self):
        self.txt_output.delete("1.0", tk.END)
        self.ax.clear()
        self.canvas.draw()
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        self.progress['value'] = 0

    # ---------------- Random Data ----------------
    # ---------------- Utility ----------------
    def open_random_dialog(self):
        d = tk.Toplevel(self.root)
        # --- Center the dialog on screen ---
        d.update_idletasks()  # Update "requested size" from geometry manager
        width = 300
        height = 150
        x = (d.winfo_screenwidth() // 2) - (width // 2)
        y = (d.winfo_screenheight() // 2) - (height // 2)
        d.geometry(f"{width}x{height}+{x}+{y}")
        d.resizable(False, False)
        d.title("Generate Random Data")
        tk.Label(d, text="Servers").grid(row=0, column=0);
        e1 = tk.Entry(d);
        e1.grid(row=0, column=1)
        tk.Label(d, text="VMs").grid(row=1, column=0);
        e2 = tk.Entry(d);
        e2.grid(row=1, column=1)
        tk.Label(d, text="Tasks").grid(row=2, column=0);
        e3 = tk.Entry(d);
        e3.grid(row=2, column=1)
        e1.insert(0, "50");
        e2.insert(0, "100");
        e3.insert(0, "30")


        def gen():
            try:
                ns = int(e1.get());
                nv = int(e2.get());
                nt = int(e3.get())
            except:
                messagebox.showerror("Invalid", "Enter valid ints")
                return
            self.servers, self.vms, self.tasks = generate_random_data(ns, nv, nt)
            # refresh trees
            self._update_server_tree()
            for i in self.tv_vms.get_children(): self.tv_vms.delete(i)
            for vm in self.vms:
                self.tv_vms.insert("", "end", values=(vm.id, vm.server_id, vm.cpu, vm.ram, vm.storage, vm.cost))
            for i in self.tv_tasks.get_children(): self.tv_tasks.delete(i)
            for t in self.tasks:
                self.tv_tasks.insert("", "end", values=(t.id, t.cpu, t.ram, t.storage, t.time, t.cost))
            self._refresh_vm_server_choices()
            self.update_server_auto_id()
            self.update_task_auto_id()

            d.destroy()
            self.update_vm_auto_id()

        ttk.Button(d, text="Generate", command=gen).grid(row=4, column=0, columnspan=2, pady=6)
        #------------- GA ----------------

    def _build_run_tab(self):

        # ======== MAIN FRAME WITH SCROLL =========
        main_container = tk.Frame(self.tab_run)
        main_container.pack(fill="both", expand=True)

        # Canvas + Scrollbar
        canvas = tk.Canvas(main_container)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame where all content goes
        frm = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frm, anchor="nw")

        # Update scroll region
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # ============================================================
        # =================== TOP CONTROL BAR =========================
        # ============================================================

        top = tk.Frame(frm)
        top.pack(fill="x", pady=6)

        btn_run = ttk.Button(top, text="Start GA", command=self.start_ga_thread)
        btn_run.pack(side="left", padx=6)

        self.progress = ttk.Progressbar(top, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side="left", padx=8)

        tk.Label(top, text="GA Time:", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        self.lbl_ga_time = tk.Label(top, text="0.0 s", font=("Arial", 10), fg="green")
        self.lbl_ga_time.pack(side="left")

        ttk.Button(top, text="Reset GA", command=self.reset_ui_GA).pack(side="right", padx=10)

        # ============================================================
        # ==================== LIVE OUTPUT (TOP) ======================
        # ============================================================

        lbl = tk.Label(frm, text="Live Output", font=("Arial", 11, "bold"))
        lbl.pack(anchor="w", padx=6)

        left = tk.Frame(frm, height=500, width=1290, bd=1, relief="solid")
        left.pack(fill="x", padx=6, pady=6)
        left.pack_propagate(False)

        output_frame = tk.Frame(left)
        output_frame.pack(fill="both", expand=True)

        self.txt_output = tk.Text(output_frame, wrap="none", height=12)
        self.txt_output.pack(side="left", fill="both", expand=True)

        scroll_y = tk.Scrollbar(output_frame, orient="vertical", command=self.txt_output.yview)
        scroll_y.pack(side="right", fill="y")
        self.txt_output.configure(yscrollcommand=scroll_y.set)
        scroll_x = tk.Scrollbar(left, orient="horizontal", command=self.txt_output.xview)
        scroll_x.pack(side="bottom", fill="x")  # هنا هياخد كل عرض left

        self.txt_output.configure(xscrollcommand=scroll_x.set)

        # ============================================================
        # =================== PLOT + LEGEND (BOTTOM) =================
        # ============================================================

        lower = tk.Frame(frm)
        lower.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Label(lower, text="Fitness Progress", font=("Arial", 11, "bold")).pack(anchor="w")

        right_inner = tk.Frame(lower)
        right_inner.pack(fill="both", expand=True)

        # --------- Plot Frame ---------
        plot_frame = tk.Frame(right_inner)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(11, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # ---------- Legend Frame ----------
        self.run_legend_outer = tk.Frame(right_inner, bd=1, relief="sunken")
        self.run_legend_outer.pack(side="right", fill="y", padx=4)

        self.run_legend_canvas = tk.Canvas(self.run_legend_outer, width=180)
        self.run_legend_canvas.pack(side="left", fill="y", expand=True)

        self.run_legend_scroll = tk.Scrollbar(self.run_legend_outer, orient="vertical",
                                              command=self.run_legend_canvas.yview)
        self.run_legend_scroll.pack(side="right", fill="y")

        self.run_legend_canvas.configure(yscrollcommand=self.run_legend_scroll.set)

        self.run_legend_inner = tk.Frame(self.run_legend_canvas)
        self.run_legend_canvas.create_window((0, 0), window=self.run_legend_inner, anchor='nw')

        self.run_legend_inner.bind(
            "<Configure>",
            lambda e: self.run_legend_canvas.configure(scrollregion=self.run_legend_canvas.bbox("all"))
        )

        # frame تحت البلوت الأساسي
        bottom_frame = tk.Frame(frm)
        bottom_frame.pack(fill="both", pady=10)

        # plot best run
        plot_best_frame = tk.Frame(bottom_frame)
        plot_best_frame.pack(side="left", fill="both", expand=True)

        self.fig_best = Figure(figsize=(4, 4))
        self.ax_best = self.fig_best.add_subplot(111)
        self.canvas_best = FigureCanvasTkAgg(self.fig_best, master=plot_best_frame)
        self.canvas_best.get_tk_widget().pack(fill="both", expand=True)

        # summary
        summary_frame = tk.Frame(bottom_frame)
        summary_frame.pack(side="right", fill="y", padx=15)

        tk.Label(summary_frame, text="Summary", font=("Arial", 11, "bold")).pack(anchor="w")
        self.summary_text = tk.Text(summary_frame, width=30, height=7)
        self.summary_text.pack()

        # Example usage:
        # resize_plot(1200, 600)  # Make plot bigger
        # resize_plot(600, 400)   # Make plot smaller

    def _clear_output_area(self):
        self.txt_output.delete("1.0", tk.END)
        self.ax.clear()
        self.canvas.draw()
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        self.progress['value'] = 0


    def start_ga_thread(self):
        # Validate data present
        if not self.tasks or not self.vms or not self.servers:
            messagebox.showerror("Missing data", "Please add at least one Server, one VM and one Task")
            return
        try:
            pop = int(self.e_pop.get())
            gens = int(self.e_gens.get())
            cpoints = int(self.e_cpoints.get())
            mrate = float(self.e_mrate.get())
            elit = int(self.e_elit.get())
            tlimit = float(self.e_time.get()) if self.e_time.get() else None
            runs = int(self.e_runs.get())
        except Exception:
            messagebox.showerror("Invalid GA params", "Enter valid GA parameters")
            return

        # disable tabs while running
        for tab in [self.tab_servers, self.tab_vms, self.tab_tasks, self.tab_ga]:
            self.nb.tab(tab, state="disabled")

        self._clear_output_area()
        self.progress['maximum'] = gens * runs  # total steps for all runs

        # run GA in thread to keep UI responsive
        t = threading.Thread(target=self._run_ga, args=(pop, gens, cpoints, mrate, elit, tlimit, runs), daemon=True)
        t.start()


    # ---------------- Thread-safe text insertion ----------------
    def _insert_colored_text(self, text, color="black", bold=False):
        """إدراج نص ملون في الـ Text widget"""
        if not hasattr(self, 'txt_output') or self.txt_output.winfo_exists() == 0:
            return

        tag_name = f"{color}_{'bold' if bold else 'normal'}"
        if tag_name not in self.txt_output.tag_names():
            self.txt_output.tag_configure(tag_name,
                                          foreground=color,
                                          font=("Arial", 10, "bold" if bold else "normal"))

        self.txt_output.insert(tk.END, text, tag_name)
        self.txt_output.see(tk.END)

    def safe_insert_text(self, text, color="black", bold=False):
        """نسخة Thread-safe لاستدعاء _insert_colored_text"""
        if hasattr(self, 'root'):
            self.root.after(0, lambda: self._insert_colored_text(text, color, bold))


    def _run_ga(self, pop, gens, cpoints, mrate, elit, tlimit, num_runs=1):
        import time

        if not hasattr(self, 'lbl_ga_time'):
            self.lbl_ga_time = tk.Label(self.tab_run, text="0.0 s", font=("Arial", 11, "bold"))
            self.lbl_ga_time.pack(side="top", pady=4)

        self.ga_all_runs_results = []
        self.ga_all_runs = []  # قائمة لكل Run تحتوي على best fitness لكل جيل

        run_colors = ["blue", "green", "red", "purple", "orange", "brown", "cyan", "magenta", "gold", "darkgreen"]

        # Initialize overall best
        #self.refresh_ga_part()
        self.overall_best_fit = float('inf')
        self.overall_best_chrom = None
        self._reset_legend()

        for run_idx in range(num_runs):
            start_time = time.time()
            self.ga_first_best_time = None
            self.ga_history = []


            gen_iter = run_ga_generator(
                self.tasks, self.vms, self.servers,
                pop_size=pop, generations=gens - 1,
                mutation_rate=mrate, crossover_points=cpoints,
                elitism=elit, time_limit=tlimit
            )

            # color = run_colors[run_idx % len(run_colors)]
            from matplotlib import cm, colors

            # لو اللون لسه ما اتعملش للـ run ده
            if run_idx not in self.run_colors:
                color_map = cm.get_cmap('tab10')
                rgba = color_map(run_idx % 10)  # اختار من colormap
                hex_color = colors.to_hex(rgba)  # تحويل إلى hex
                self.run_colors[run_idx] = hex_color  # حفظ اللون

            # استخدم اللون هنا
            color = self.run_colors[run_idx]

            self._insert_colored_text(f"\n{'=' * 190}\n  STARTING RUN  {run_idx + 1}\n{'=' * 190}\n", color, bold=True)

            for gen_idx, best_chrom, best_fit, history in gen_iter:
                elapsed = time.time() - start_time

                # الوقت لأول أفضل حل
                if self.ga_first_best_time is None or best_fit < min(self.ga_history, default=float('inf')):
                    self.ga_first_best_time = elapsed

                # Update GA time label
                self.root.after(0, lambda t=elapsed, r=run_idx: self.lbl_ga_time.config(
                    text=f"Run {r + 1}, {t:.2f}s"))

                # Live Output
                gen_text = f"Run {run_idx + 1}, Gen {gen_idx + 1} | Best fitness: {best_fit:.4f} | Elapsed: {elapsed:.2f}s\n"
                gen_text += f"Best chromosome: {best_chrom}\n"
                gen_text += "\n".join([f" Task {t} -> VM {vm}" for t, vm in enumerate(best_chrom)]) + "\n\n"
                self._insert_colored_text(gen_text, color)

                # Update plot
                self.root.after(0, lambda h=history, r=run_idx: self._update_plot(h, run_idx=r))

                # Update progress bar
                if hasattr(self, 'progress'):
                    try:
                        self.root.after(0, lambda: self.progress.step(1))
                    except tk.TclError:
                        pass

                time.sleep(0.02)
                self.ga_history.append(best_fit)

            # End of run: save results
            final_best = best_chrom
            final_fit = best_fit
            total_elapsed = time.time() - start_time
            self.ga_all_runs_results.append((final_best, final_fit, total_elapsed))
            self.ga_all_runs.append(self.ga_history.copy())

            # Update overall best
            if final_fit < self.overall_best_fit:
                self.overall_best_fit = final_fit
                self.overall_best_chrom = final_best

            # Summary for this run
            summary_text = f"\n{'-' * 90}\n  SUMMARY RUN  {run_idx + 1}\n\n"
            summary_text += f"  Best Fitness: {final_fit:.4f}\n"
            summary_text += f"  Best Chromosome: {final_best}\n"
            summary_text += f"  First Best Found At: {self.ga_first_best_time:.2f}s\n"
            summary_text += f"  Total Run Time: {total_elapsed:.2f}s\n"
            summary_text += f"{'-' * 90}\n"
            self._insert_colored_text(summary_text, color, bold=True)

            # Update comparison and GA time label
            self.root.after(0, self.update_comparison)
            self.root.after(0,
                            lambda t=total_elapsed, r=run_idx: self.lbl_ga_time.config(
                                text=f"Run {r + 1} finished: {t:.2f}s"))

        # After all runs: show summary for the run with lowest fitness
        self.root.after(0, lambda: self._show_summary_tab(self.overall_best_chrom, self.overall_best_fit))
        self._plot_best_run()

        # Re-enable tabs
        for tab in [self.tab_servers, self.tab_vms, self.tab_tasks, self.tab_ga]:
            self.root.after(0, lambda t=tab: self.nb.tab(t, state="normal"))


    import numpy as np
    from matplotlib import cm

    def _update_plot(self, history, run_idx=0):
        if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return

        # اختيار اللون لكل run
        # color_map = cm.get_cmap('tab10')
        # color = color_map(run_idx % 10)
        color = self.run_colors[run_idx % len(self.run_colors)]
        color_map = cm.get_cmap('tab10')
        color = color_map(run_idx % 10)

        # حوّل اللون لـ hex مرة واحدة
        from matplotlib import colors
        hex_color = colors.to_hex(color)

        # ارسم الخط باللون
        #self.ax.plot(history, color=hex_color, label=f"Run {run_idx + 1}")

        gens = np.arange(1, len(history) + 1)

        # رسم الخط بدون markers ثقيلة
        line, = self.ax.plot(
            gens, history, linestyle='-', color=color,
            alpha=0.8, label=f'Run {run_idx + 1}'
        )
        # تأكد من وجود set لتخزين runs المضافة
        if not hasattr(self, '_runs_added'):
            self._runs_added = set()


        if run_idx not in self._runs_added:
            # استخدم hex_color اللي جه من update_plot
            legend_item = tk.Frame(self.run_legend_inner)
            legend_item.pack(fill="x", pady=2)

            color_box = tk.Label(legend_item, bg=hex_color, width=2, height=1)
            color_box.pack(side="left", padx=2)

            tk.Label(
                legend_item,
                text=f"Run {run_idx + 1}",
                anchor="w",
                font=("Arial", 10, "bold")
            ).pack(side="left", padx=4)

            # سجّل إنه اتضاف
            self._runs_added.add(run_idx)

        # Labels و grid
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Fitness")
        self.ax.set_title("GA Fitness Progress Across Runs")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # Legend: unique labels فقط
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        #self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

        # إعداد annotation مرة واحدة
        if not hasattr(self, 'annot'):
            self.annot = self.ax.annotate(
                "", xy=(0, 0), xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color='black')
            )
            self.annot.set_visible(False)

        # تخزين نقاط hover المهمة فقط
        # هنا نركز على أقل قيمة لكل run + كل 10 نقاط لتخفيف الضغط
        if not hasattr(self, '_hover_points'):
            self._hover_points = []

        min_idx = np.argmin(history)
        self._hover_points.append(
            {'x': gens[min_idx], 'y': history[min_idx], 'color': color, 'label': f'Run {run_idx + 1}'})

        # إضافة نقاط كل N نقطة لتخفيف الحمل
        N = max(1, len(history) // 100)  # لو البيانات كبيرة، ناخد كل 100 نقطة مثلاً
        for i in range(0, len(history), N):
            self._hover_points.append({'x': gens[i], 'y': history[i], 'color': color, 'label': f'Run {run_idx + 1}'})

        # Hover function سريع جدًا على نقاط محددة
        def hover(event):
            if event.inaxes != self.ax or event.x is None or event.y is None:
                return

            visible = False
            mouse_xy = np.array([event.x, event.y])
            for point in self._hover_points:
                x_pixel, y_pixel = self.ax.transData.transform((point['x'], point['y']))
                distance = np.linalg.norm(np.array([x_pixel, y_pixel]) - mouse_xy)
                if distance < 10:  # threshold 10 pixels
                    self.annot.xy = (point['x'], point['y'])
                    self.annot.set_text(f"{point['label']}\nGen: {int(point['x'])}\nFitness: {point['y']:.4f}")
                    self.annot.get_bbox_patch().set_facecolor(point['color'])
                    self.annot.get_bbox_patch().set_alpha(0.8)
                    self.annot.set_visible(True)
                    visible = True
                    break

            if not visible:
                self.annot.set_visible(False)

            self.canvas.draw_idle()

        # Connect hover مرة واحدة فقط
        if not hasattr(self, '_hover_connected'):
            self.canvas.mpl_connect("motion_notify_event", hover)
            self._hover_connected = True

        self.canvas.draw()
        # حفظ history الخاص بكل Run
        if not hasattr(self, "run_histories"):
            self.run_histories = {}

        self.run_histories[run_idx] = history[:]

    def _reset_legend(self):
        # امسح كل عناصر الـ legend
        for w in self.run_legend_inner.winfo_children():
            w.destroy()

        # صفّي المجموعة
        self._runs_added.clear()

    def _compute_best_run(self):
        global_min = float("inf")
        best_run = -1
        best_gen = -1

        for run_idx, history in self.run_histories.items():
            local_min = min(history)
            gen_idx = history.index(local_min)

            if local_min < global_min:
                global_min = local_min
                best_run = run_idx
                best_gen = gen_idx

        return best_run, best_gen, global_min


    def _plot_best_run(self):
        best_run, best_gen, global_min = self._compute_best_run()
        history = self.run_histories[best_run]

        self.ax_best.clear()

        # نفس اللون المستخدم في الرسم الكبير
        color = self.run_colors[best_run]

        # Plot line
        self.ax_best.plot(
            range(1, len(history) + 1),
            history,
            linewidth=2,
            color=color
        )

        self.ax_best.set_title(f"Best Run (Run {best_run + 1})")
        self.ax_best.set_xlabel("Generation")
        self.ax_best.set_ylabel("Fitness")

        # Scatter لنفس اللون
        self.ax_best.scatter(best_gen + 1, global_min, s=50, color=color)

        # Annotation بنفس اللون
        self.ax_best.annotate(
            f"Min={global_min}",
            (best_gen + 1, global_min),
            textcoords="offset points",
            xytext=(10, -10),
            color=color,
            fontsize=10,
            fontweight='bold'
        )

        self.canvas_best.draw()

        # Summary
        self.summary_text.delete("1.0", "end")

        self.summary_text.tag_config("title", font=("Arial", 14, "bold"), foreground="blue")
        self.summary_text.tag_config("key", font=("Arial", 12, "bold"), foreground="black")
        self.summary_text.tag_config("value", font=("Arial", 12), foreground="black")

        self.summary_text.insert("end", "=== Summary ===\n", "title")
        self.summary_text.insert("end", "Lowest Fitness: ", "key")
        self.summary_text.insert("end", f"{global_min}\n", "value")
        self.summary_text.insert("end", "Found in Run: ", "key")
        self.summary_text.insert("end", f"{best_run + 1}\n", "value")
        self.summary_text.insert("end", "Generation: ", "key")
        self.summary_text.insert("end", f"{best_gen + 1}\n", "value")

    def _show_summary1(self, best_chrom, best_fit):
        # clear summary frame
        for w in self.summary_frame.winfo_children():
            w.destroy()
        tk.Label(self.summary_frame, text=f"Final Fitness: {best_fit:.2f}", font=("Arial", 11, "bold")).pack(anchor="w")
        tk.Label(self.summary_frame, text=f"Best Chromosome: {best_chrom}").pack(anchor="w")




    def _show_summary_tab(self, best_chrom, best_fit):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import tkinter as tk
        from tkinter import ttk

        # ---------------- CLEAR OLD CONTENT ----------------
        for w in self.summary_frame.winfo_children():
            w.destroy()


        # ---------------- TITLE SECTION ----------------
        tk.Label(self.summary_frame, text=f"Final Fitness: {best_fit:.2f}",
                 font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)

        # Frame للكـروموسوم
        chrom_frame = tk.Frame(self.summary_frame)
        chrom_frame.pack(anchor="w", padx=10, pady=(0, 10))

        # نص البداية
        tk.Label(chrom_frame, text="Best Chromosome:", font=("Arial", 12)).pack(side="left")


        # ---------- Scrollable Chromosome ----------
        canvas = tk.Canvas(self.summary_frame, height=40)
        canvas.pack(side="top", fill="x", expand=True)

        h_scroll = tk.Scrollbar(self.summary_frame, orient="horizontal", command=canvas.xview)
        h_scroll.pack(side="bottom", fill="x")

        canvas.configure(xscrollcommand=h_scroll.set)

        chrom_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=chrom_frame, anchor="nw")

        # Tooltip class
        class Tooltip:
            def __init__(self, widget, text):
                self.widget = widget
                self.text = text
                self.tip = None
                self.widget.bind("<Enter>", self.show)
                self.widget.bind("<Leave>", self.hide)

            def show(self, event=None):
                if self.tip or not self.text:
                    return
                x, y, cx, cy = self.widget.bbox("insert") or (0, 0, 0, 0)
                x += self.widget.winfo_rootx() + 25
                y += self.widget.winfo_rooty() + 20
                self.tip = tk.Toplevel(self.widget)
                self.tip.wm_overrideredirect(True)
                self.tip.wm_geometry(f"+{x}+{y}")
                label = tk.Label(self.tip, text=self.text, background="lightyellow",
                                 relief="solid", borderwidth=1, font=("Arial", 10))
                label.pack()

            def hide(self, event=None):
                if self.tip:
                    self.tip.destroy()
                    self.tip = None

        # عرض القيم داخل الأقواس مع Tooltip لكل رقم
        tk.Label(chrom_frame, text="[", font=("Arial", 12)).pack(side="left")

        for i, vm in enumerate(best_chrom):
            lbl = tk.Label(chrom_frame, text=str(vm), font=("Arial", 12))
            lbl.pack(side="left")
            Tooltip(lbl, text=f"Task {i} on VM {vm}")
            if i < len(best_chrom) - 1:
                tk.Label(chrom_frame, text=", ", font=("Arial", 12)).pack(side="left")

        tk.Label(chrom_frame, text="]", font=("Arial", 12)).pack(side="left")

        # تحديث scrollregion
        chrom_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # ---------------- CREATE NOTEBOOK ----------------
        summary_notebook = ttk.Notebook(self.summary_frame)
        summary_notebook.pack(fill="both", expand=True)

        tab_chrom = tk.Frame(summary_notebook)
        tab_vm = tk.Frame(summary_notebook)
        tab_server = tk.Frame(summary_notebook)

        summary_notebook.add(tab_chrom, text="Chromosome Mapping")
        # summary_notebook.add(tab_vm, text="VM Charts")
        # summary_notebook.add(tab_server, text="Server Charts")

        # ---------------- CHROMOSOME SCROLL SETUP ----------------
        chrom_canvas = tk.Canvas(tab_chrom, bg="#ffffff", height=350)
        chrom_canvas.pack(side="top", fill="both", expand=True)

        # Scrollbar أفقي
        chrom_scroll_x = tk.Scrollbar(tab_chrom, orient="horizontal", command=chrom_canvas.xview)
        chrom_scroll_x.pack(side="bottom", fill="x")

        # Scrollbar عمودي
        chrom_scroll_y = tk.Scrollbar(tab_chrom, orient="vertical", command=chrom_canvas.yview)
        chrom_scroll_y.pack(side="right", fill="y")

        # ربط الـ Canvas بالـ Scrollbars
        chrom_canvas.configure(xscrollcommand=chrom_scroll_x.set,
                               yscrollcommand=chrom_scroll_y.set)

        # Frame داخل الـ Canvas
        chrom_frame = tk.Frame(chrom_canvas, bg="#ffffff")
        chrom_canvas.create_window((0, 0), window=chrom_frame, anchor="nw")

        # تحديث المساحة المتاحة للسكرول تلقائياً
        chrom_frame.bind("<Configure>", lambda e: chrom_canvas.configure(
            scrollregion=chrom_canvas.bbox("all")
        ))

        # ---------------- MAP CHROMOSOME ----------------
        vm_ids = [vm.id for vm in self.vms]
        vm_tasks = defaultdict(list)

        for task_idx, vm_idx in enumerate(best_chrom):
            if 0 <= vm_idx < len(vm_ids):
                actual_vm_id = vm_ids[vm_idx]
                vm_tasks[actual_vm_id].append(task_idx)

        servers_frame = tk.Frame(chrom_frame)
        servers_frame.pack(fill="both", expand=True, pady=10)

        for server in self.servers:
            server_frame = tk.LabelFrame(servers_frame, text=f"Server {server.id}",
                                         padx=10, pady=10, bg="#d0e1f9", font=("Arial", 10, "bold"))
            server_frame.pack(side="left", padx=10, pady=10, fill="y")

            for vm in server.vms:
                vm_frame = tk.LabelFrame(server_frame, text=f"VM {vm.id}",
                                         padx=5, pady=5, bg="#b2f7b2")
                vm_frame.pack(pady=5, fill="both")

                for t in vm_tasks.get(vm.id, []):
                    tk.Label(vm_frame, text=f"Task {t}", bg="#f7f7b2").pack(padx=2, pady=2, anchor="w")

    def load_dataset(self):
        import pandas as pd
        from tkinter import filedialog, messagebox

        file_path = filedialog.askopenfilename(
            title="Choose Dataset CSV",
            filetypes=[("CSV Files", "*.csv")]
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)

            # Required columns
            required = ["cpu", "ram", "storage", "cost"]
            for col in required:
                if col not in df.columns:
                    messagebox.showerror("Error", f"Missing column: {col}")
                    return

            added = 0
            for _, row in df.iterrows():
                task_id = len(self.tasks) + 1
                cpu = int(row["cpu"])
                ram = int(row["ram"])
                storage = int(row["storage"])
                cost = float(row["cost"])

                # Create Task object
                task = Task(task_id, cpu, ram, storage, cost)

                self.tasks.append(task)
                self.tv_tasks.insert("", "end",
                                     values=(task.id, task.cpu, task.ram, task.storage, task.time, task.cost))

                added += 1

            messagebox.showinfo("Success", f"Loaded {added} tasks!")

            self._update_task_graph()
            self.update_task_auto_id()

        except Exception as e:
            messagebox.showerror("Error", str(e))


    def _build_de_params_tab(self):
        frm = tk.Frame(self.tab_de_params)
        frm.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Label(frm, text="Differential Evolution Parameters", font=("Arial", 14, "bold")).pack(pady=10)

        f = tk.Frame(frm)
        f.pack(pady=10)

        labels = ["Population Size", "Generations", "Crossover Rate (CR)", "Differential Weight (F)", "Number of Runs"]
        for i, text in enumerate(labels):
            tk.Label(f, text=text).grid(row=i, column=0, sticky="w")

        self.e_de_pop = tk.Entry(f, width=10)
        self.e_de_pop.insert(0, "100")
        self.e_de_gen = tk.Entry(f, width=10)
        self.e_de_gen.insert(0, "100")
        self.e_de_cr = tk.Entry(f, width=10)
        self.e_de_cr.insert(0, "0.9")
        self.e_de_f = tk.Entry(f, width=10)
        self.e_de_f.insert(0, "0.8")
        self.e_de_runs = tk.Entry(f, width=10)
        self.e_de_runs.insert(0, "1")

        entries = [self.e_de_pop, self.e_de_gen, self.e_de_cr, self.e_de_f, self.e_de_runs]
        for i, entry in enumerate(entries):
            entry.grid(row=i, column=1, padx=5, pady=4)

        ttk.Button(frm, text="Run DE in Run tab", command=self.start_de_and_switch).pack(pady=10)

    # def start_de_and_switch(self):
    #     try:
    #         pop = int(self.e_de_pop.get())
    #         gens = int(self.e_de_gen.get())
    #         cr = float(self.e_de_cr.get())
    #         f_weight = float(self.e_de_f.get())
    #         num_runs = int(self.e_de_runs.get()) if self.e_de_runs.get() else 1
    #     except ValueError:
    #         messagebox.showerror("Error", "Please enter valid DE parameters!")
    #         return
    #
    #     self.de_params = {"pop_size": pop, "generations": gens, "cr": cr, "f_weight": f_weight, "num_runs": num_runs}
    #
    #     # Switch to results tab
    #     self.nb.select(self.tab_de_results)
    #
    #     # Run DE in a separate thread to avoid freezing UI
    #     threading.Thread(target=self.run_de, args=(pop, gens, cr, f_weight, num_runs), daemon=True).start()
    def start_de_and_switch(self):
        try:
            pop = int(self.e_de_pop.get())
            gens = int(self.e_de_gen.get())
            cr = float(self.e_de_cr.get())
            f_weight = float(self.e_de_f.get())
            num_runs = int(self.e_de_runs.get()) if self.e_de_runs.get() else 1
        except ValueError:
            messagebox.showerror("Error", "Please enter valid DE parameters!")
            return

        # حفظ القيم فقط
        self.de_params = {
            "pop_size": pop,
            "generations": gens,
            "cr": cr,
            "f_weight": f_weight,
            "num_runs": num_runs
        }

        # بدون تشغيل — فقط سويتش
        self.nb.select(self.tab_de_results)


    def _build_de_results_tab(self):
        # --- Canvas خارجي للـ scrolling ---
        canvas = tk.Canvas(self.tab_de_results)
        canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar عمودي للـ canvas
        v_scroll = tk.Scrollbar(self.tab_de_results, orient="vertical", command=canvas.yview)
        v_scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=v_scroll.set)


        # Frame المحتوى الحقيقي داخل الـ canvas
        frm = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frm, anchor="nw")

        # تحديث scrollregion تلقائياً عند تغير حجم المحتوى
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # ----------------- Top controls -----------------
        top_frame = tk.Frame(frm)
        top_frame.pack(side="top", fill="x", pady=5, padx=5)
        ttk.Button(top_frame, text="Start DE", command=lambda: self.start_de_thread()).pack(side="left", padx=6)
        ttk.Button(top_frame, text="Reset DE", command=self.reset_ui_DE).pack(side="right", padx=6)

        self.progress_de = ttk.Progressbar(top_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_de.pack(side="left", padx=6)

        tk.Label(top_frame, text="DE Time:", font=("Arial", 10, "bold")).pack(side="left", padx=6)
        self.lbl_de_time = tk.Label(top_frame, text="0.0 s", font=("Arial", 10), fg="green")
        self.lbl_de_time.pack(side="left", padx=6)

        # ----------------- Live Output -----------------
        tk.Label(frm, text="Live Output", font=("Arial", 11, "bold")).pack(anchor="w")
        output_frame = tk.Frame(frm)
        output_frame.pack(fill="both", expand=False, pady=5)

        self.txt_de_output = tk.Text(output_frame, wrap="none", height=30 ,width=160)
        self.txt_de_output.pack(side="left", fill="both", expand=True)
        scroll_y = tk.Scrollbar(output_frame, orient="vertical", command=self.txt_de_output.yview)
        scroll_y.pack(side="right", fill="y")
        self.txt_de_output.configure(yscrollcommand=scroll_y.set)
        scroll_x = tk.Scrollbar(frm, orient="horizontal", command=self.txt_de_output.xview)
        scroll_x.pack(side="top", fill="x")
        self.txt_de_output.configure(xscrollcommand=scroll_x.set)

        # ----------------- Plot + Legend -----------------
        plot_legend_frame = tk.Frame(frm)
        plot_legend_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: Main Plot
        plot_frame = tk.Frame(plot_legend_frame)
        plot_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.de_fig = Figure(figsize=(9, 6))
        self.de_ax = self.de_fig.add_subplot(111)

        self.de_canvas = FigureCanvasTkAgg(self.de_fig, master=plot_frame)
        self.de_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Right: Legend
        self.de_legend_outer = tk.Frame(plot_legend_frame, bd=1, relief="sunken", width=150, height=400)
        self.de_legend_outer.pack(side="right", fill="y", padx=4, pady=4)
        self.de_legend_outer.pack_propagate(False)

        # استخدم grid للـ canvas + scrollbar لضمان ظهور scrollbar دائم
        self.de_legend_canvas = tk.Canvas(self.de_legend_outer)
        self.de_legend_canvas.grid(row=0, column=0, sticky="nsew")
        self.de_legend_outer.grid_rowconfigure(0, weight=1)
        self.de_legend_outer.grid_columnconfigure(0, weight=1)

        self.de_legend_scroll = tk.Scrollbar(self.de_legend_outer, orient="vertical",
                                             command=self.de_legend_canvas.yview)
        self.de_legend_scroll.grid(row=0, column=1, sticky="ns")

        self.de_legend_canvas.configure(yscrollcommand=self.de_legend_scroll.set)

        # Inner frame داخل الـ canvas
        self.de_legend_inner = tk.Frame(self.de_legend_canvas)
        self.de_legend_canvas.create_window((0, 0), window=self.de_legend_inner, anchor="nw")

        # تحديث scrollregion تلقائي عند تغير المحتوى
        self.de_legend_inner.bind(
            "<Configure>",
            lambda e: self.de_legend_canvas.configure(scrollregion=self.de_legend_canvas.bbox("all"))
        )

        # Force scrollbar visible دائمًا بإضافة padding صغير إذا محتوى قليل
        self.de_legend_inner.update_idletasks()
        if self.de_legend_canvas.bbox("all")[3] < 400:
            self.de_legend_canvas.config(scrollregion=(0, 0, 200, 400))

        # ----------------- Summary frame -----------------
        self.de_summary_frame = tk.Frame(frm)
        self.de_summary_frame.pack(fill="x", pady=6)

        # ----------------- Best Run Plot + Summary (SIDE BY SIDE) -----------------
        tk.Label(frm, text="Best Run Convergence", font=("Arial", 11, "bold")).pack(anchor="w", pady=4)

        # Frame يحتوي الرسم + الـ Summary
        best_plot_frame = tk.Frame(frm)
        best_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # ---------------------------------
        # Left: Best Run Plot
        # ---------------------------------
        self.best_run_plot_frame = tk.Frame(best_plot_frame)
        self.best_run_plot_frame.pack(side="left", fill="both", expand=True)

        self.de_fig_best = Figure(figsize=(6, 5))
        self.de_ax_best = self.de_fig_best.add_subplot(111)

        self.de_canvas_best = FigureCanvasTkAgg(self.de_fig_best, master=self.best_run_plot_frame)
        self.de_canvas_best.get_tk_widget().pack(fill="both", expand=True)

        # ---------------------------------
        # Right: Best Run Summary
        # ---------------------------------
        self.best_run_summary_frame = tk.Frame(best_plot_frame, bg="#f0f0f0", width=260)
        self.best_run_summary_frame.pack(side="right", fill="y", padx=5)

        tk.Label(
            self.best_run_summary_frame, text="Best Run Summary",
            font=("Arial", 12, "bold"), bg="#f0f0f0"
        ).pack(pady=10)

        self.txt_best_run_summary = tk.Text(
            self.best_run_summary_frame,
            width=32,
            height=20,
            font=("Consolas", 10),
            bg="white"
        )
        self.txt_best_run_summary.pack(padx=8, pady=5)

    # ------------------ DE Thread ------------------
    def start_de_thread(self):
        threading.Thread(target=self.run_de, daemon=True).start()


    def run_de(self, pop=None, gens=None, CR=None, F=None, num_runs=None):
        import numpy as np
        from copy import deepcopy
        import time

        # قراءة القيم من المدخلات إذا لم تُمرر
        if not (pop and gens and CR and F and num_runs):
            pop = int(self.e_de_pop.get())
            gens = int(self.e_de_gen.get())
            CR = float(self.e_de_cr.get())
            F = float(self.e_de_f.get())
            num_runs = int(self.e_de_runs.get()) if self.e_de_runs.get() else 1

        # تحقق من وجود VMs وTasks
        if not self.vms or not self.tasks:
            messagebox.showwarning("Warning", "Please load VMs and Tasks first!")
            return

        # مسح النصوص القديمة والرسم عند إعادة التشغيل
        #self.refresh_de_part()
        self.txt_de_output.delete("1.0", "end")
        self.de_ax.clear()
        if hasattr(self, '_runs_added'):
            self._runs_added.clear()
        if hasattr(self, '_hover_points'):
            self._hover_points.clear()
        self._runs_added = set()
        self._hover_points = []

        best_of_all = None
        best_fit_of_all = float('inf')
        all_histories = []
        self.de_all_runs = []  # قائمة لكل Run تحتوي على best fitness لكل جيل

        self.de_running = True
        self.de_start_time = time.time()
        self.progress_de["maximum"] = num_runs
        self.progress_de["value"] = 0

        self.update_de_timer()  # Start live timer

        # ألوان مختلفة لكل Run
        run_colors = ["blue", "green", "red", "purple", "orange", "brown", "cyan", "magenta", "darkgreen", "darkred"]

        # إعداد annotation للـ hover
        if not hasattr(self, 'annot'):
            self.annot = self.de_ax.annotate(
                "", xy=(0, 0), xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color='black')
            )
            self.annot.set_visible(False)

        # Hover function
        def hover(event):
            if event.inaxes != self.de_ax:
                self.annot.set_visible(False)
                self.de_canvas.draw_idle()
                return
            visible = False
            mouse_xy = np.array([event.x, event.y])
            for point in self._hover_points:
                x_pixel, y_pixel = self.de_ax.transData.transform((point['x'], point['y']))
                if np.linalg.norm(np.array([x_pixel, y_pixel]) - mouse_xy) < 10:
                    self.annot.xy = (point['x'], point['y'])
                    self.annot.set_text(f"{point['label']}\nGen:{int(point['x'])}\nFitness:{point['y']:.4f}")
                    self.annot.get_bbox_patch().set_facecolor(point['color'])
                    self.annot.get_bbox_patch().set_alpha(0.8)
                    self.annot.set_visible(True)
                    visible = True
                    break
            if not visible:
                self.annot.set_visible(False)
            self.de_canvas.draw_idle()

        if not hasattr(self, '_hover_connected'):
            self.de_canvas.mpl_connect("motion_notify_event", hover)
            self._hover_connected = True

        # بدء كل Run
        for run_idx in range(num_runs):

            de = DifferentialEvolution(
                vms=self.vms, tasks=self.tasks, servers=self.servers,
                pop_size=pop, generations=gens, F=F, CR=CR
            )

            history = []
            best = None
            color_tag = f"run_{run_idx}"
            self.txt_de_output.tag_config(color_tag, foreground=run_colors[run_idx % len(run_colors)])
            # قبل الحلقة
            self.de_first_best_time = 0
            best_fit_so_far = float('inf')
            start_time = time.time()
            # كل جيل
            for gen in range(int(gens)):
                new_pop = []
                for i in range(int(pop)):
                    mutant = de.mutate(i)
                    trial = de.crossover(de.population[i], mutant)
                    selected = de.select(de.population[i], trial)
                    new_pop.append(selected)

                de.population = np.array(new_pop)
                fitness_values = [de.fitness(ind) for ind in de.population]
                best_idx = np.argmin(fitness_values)
                best_in_gen = deepcopy(de.population[best_idx])
                best_fit_in_gen = fitness_values[best_idx]
                if best_fit_in_gen < best_fit_so_far:
                    best_fit_so_far = best_fit_in_gen
                    # تسجيل الوقت عند أول أفضل قيمة
                    self.de_first_best_time = time.time() - start_time
                    best = best_in_gen

                if best is None or best_fit_in_gen < de.fitness(best):
                    best = best_in_gen

                history.append(best_fit_in_gen)

                # عرض نصي مباشر لكل جيل
                self.txt_de_output.insert(
                    "end", f"Run {run_idx + 1} ---->  Gen {gen + 1}  ----> Best Fitness = {best_fit_in_gen:.4f}\n",
                    color_tag
                )
                self.txt_de_output.see("end")
                self.root.update_idletasks()

            # تحديث الرسم لكل Run بعد الانتهاء منه
            all_histories.append(history)
            self.de_all_runs.append(history.copy())
            self._hover_points.clear()  # مسح نقاط hover القديمة
            self.de_ax.clear()
            for widget in self.de_legend_inner.winfo_children(): widget.destroy()
            self._runs_added.clear()
            self._hover_points.clear()

            for i, hist in enumerate(all_histories):
                color = run_colors[i % len(run_colors)]
                # self.de_ax.plot(hist, linestyle='-', marker=None, color=color, label=f'Run {i + 1}')
                self.de_ax.plot(range(1, len(hist) + 1), hist, linestyle='-', marker=None, color=color,
                                label=f'Run {i + 1}')

                # Legend Tkinter
                if i not in self._runs_added and hasattr(self, 'de_legend_inner'):
                    self._runs_added.add(i)
                    legend_item = tk.Frame(self.de_legend_inner)
                    legend_item.pack(fill="x", pady=2)
                    tk.Label(legend_item, bg=color, width=2, height=1).pack(side="left", padx=2)
                    tk.Label(legend_item, text=f"Run {i + 1}", font=("Arial", 10, "bold")).pack(side="left", padx=4)

                # أقل نقطة لكل Run للـ hover
                min_idx = np.argmin(hist)
                self._hover_points.append({'x': min_idx+1, 'y': hist[min_idx], 'color': color, 'label': f'Run {i + 1}'})

            self.de_ax.set_title("DE Convergence")
            self.de_ax.set_xlabel("Generation")
            self.de_ax.set_ylabel("Best Fitness")
            self.de_ax.grid(True, linestyle='--', alpha=0.6)
            #self.de_ax.legend(loc='upper right', fontsize=9)
            self.de_canvas.draw()

            # تحديث أفضل قيمة عبر كل Runs
            best_cost = de.fitness(best)
            if best_cost < best_fit_of_all:
                best_fit_of_all = best_cost
                best_of_all = best
                best_history = history

            self.progress_de["value"] = run_idx + 1

            # فاصل ونص بلون Run
            self.txt_de_output.insert("end", "=" * 100 + "\n", color_tag)
            self.txt_de_output.insert("end", f"Run {run_idx + 1} finished: Best Fitness = {best_cost:.4f}\n", color_tag)
            self.txt_de_output.insert("end", "=" * 100 + "\n\n\n", color_tag)
            self.txt_de_output.see("end")
            # ---- رسم أفضل Run بدون نقاط ----
            best_run_idx = np.argmin([min(h) for h in all_histories])
            best_run_history = all_histories[best_run_idx]
            best_color = run_colors[best_run_idx % len(run_colors)]

            # رسم الخط بس، بدون marker
            line_best, = self.de_ax_best.plot(range(1, len(best_run_history)+1), best_run_history, linestyle='-' , color=best_color, label=f'Best Run {best_run_idx + 1}')


            self.best_run_idx = np.argmin([min(h) for h in all_histories])

            # إعداد hover لكل نقطة على الخط
            import mplcursors
            cursor = mplcursors.cursor(line_best, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                gen = int(sel.target[0]) + 1
                val = sel.target[1]
                sel.annotation.set_text(f"Run {best_run_idx + 1}\nGen: {gen}\nFitness: {val:.4f}")
                sel.annotation.get_bbox_patch().set_facecolor(best_color)
                sel.annotation.get_bbox_patch().set_alpha(0.8)

            self.de_canvas.draw()

            self.de_ax_best.clear()
            self.de_ax_best.plot(best_run_history, linestyle='-' ,
                                 color=best_color, label=f'Best Run {best_run_idx + 1}')

            self.de_ax_best.set_title(f"Best Run: Run {best_run_idx + 1}")
            self.de_ax_best.set_xlabel("Generation")
            self.de_ax_best.set_ylabel("Best Fitness")
            self.de_ax_best.grid(True, linestyle='--', alpha=0.6)

            self.de_canvas_best.draw()
            # تعبئة الـ Summary
            self.txt_best_run_summary.delete("1.0", "end")

            best_fitness = min(best_run_history)
            best_gen_idx = best_run_history.index(best_fitness)
            self.de_history = history
            self.update_comparison()

            # self.txt_best_run_summary.insert("end", f"Lowest Fitness : {best_fitness:.4f}\n")
            # self.txt_best_run_summary.insert("end", f"Found in Run : {best_run_idx + 1}\n")
            # self.txt_best_run_summary.insert("end", f"Generation  : {best_gen_idx + 1}\n\n")
            # تنظيف النص
            self.txt_best_run_summary.delete("1.0", "end")

            # تعريف ستايلات
            self.txt_best_run_summary.tag_config("key", font=("Arial", 12, "bold"), foreground="black")
            self.txt_best_run_summary.tag_config("value", font=("Arial", 12), foreground="black")
            self.txt_best_run_summary.tag_config("title", font=("Arial", 14, "bold"), foreground="blue")

            # كتابة العنوان
            self.txt_best_run_summary.insert("end", "=== Summary ===\n", "title")

            # كتابة البيانات مع ستايل
            self.txt_best_run_summary.insert("end", "Lowest Fitness : ", "key")
            self.txt_best_run_summary.insert("end", f"{best_fitness:.4f}\n", "value")

            self.txt_best_run_summary.insert("end", "Found in Run : ", "key")
            self.txt_best_run_summary.insert("end", f"{best_run_idx + 1}\n", "value")

            self.txt_best_run_summary.insert("end", "Generation  : ", "key")
            self.txt_best_run_summary.insert("end", f"{best_gen_idx + 1}\n\n", "value")

        self.de_running = False
        elapsed_time = time.time() - self.de_start_time
        self.lbl_de_time.config(text=f"{elapsed_time:.2f} s")
        self.show_de_mapping(best_of_all,best_fit_of_all)
        #messagebox.showinfo("DE Finished", f"Differential Evolution Finished!\nBest Fitness = {best_fit_of_all:.4f}")

    # ------------------ Timer ------------------
    def update_de_timer(self):
        if self.de_running:
            elapsed = time.time() - self.de_start_time
            self.lbl_de_time.config(text=f"{elapsed:.2f} s")
            self.root.after(100, self.update_de_timer)


    def show_de_mapping(self, best_solution , best_fitness):
        """
        عرض الماب الخاصة بأفضل Run في DE
        best_solution: أفضل فرد (VM assignment array)
        """
        from collections import defaultdict

        # حذف أي محتوى قديم
        for widget in self.tab_de_mapping.winfo_children():
            widget.destroy()

        # عنوان رئيسي
        tk.Label(
            self.tab_de_mapping,
            text="Best Run Task Mapping",
            font=("Arial", 13, "bold"),
            fg="darkblue"
        ).pack(pady=8)
        # ----- عرض أفضل Fitness -----
        tk.Label(
            self.tab_de_mapping,
            text=f"Best Fitness: {best_fitness:.4f}" if isinstance(best_fitness,
                                                                   float) else f"Best Fitness: {best_fitness}",
            font=("Arial", 14, "bold"),  # حجم أكبر ونص Bold
            fg="black",  # اللون أسود
            padx=10, pady=5
        ).pack(side="top", anchor="w", padx=10, pady=8)

        # استخراج IDs
        vm_ids = [vm.id for vm in self.vms]
        vm_tasks = defaultdict(list)

        # ربط كل Task بالـ VM الخاص بها
        for t_idx, vm_idx in enumerate(best_solution):
            safe_vm_index = int(round(vm_idx))  # تأمين الفهرس
            if 0 <= safe_vm_index < len(vm_ids):
                vm_id = vm_ids[safe_vm_index]
                vm_tasks[vm_id].append(self.tasks[t_idx].id)

        # ----------------- بناء UI -----------------

        outer_container = tk.Frame(self.tab_de_mapping)
        outer_container.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer_container, bg="#eef", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar_y = ttk.Scrollbar(outer_container, orient="vertical", command=canvas.yview)
        scrollbar_y.pack(side="right", fill="y")

        scrollbar_x = ttk.Scrollbar(self.tab_de_mapping, orient="horizontal", command=canvas.xview)
        scrollbar_x.pack(side="bottom", fill="x")

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        inner_frame = tk.Frame(canvas, bg="#eef")
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # ----------------- عرض الخريطة -----------------
        col = 0
        for server in self.servers:
            server_frame = tk.LabelFrame(
                inner_frame,
                text=f"Server {server.id}",
                padx=10,
                pady=10,
                bg="#d0e1f9",
                font=("Arial", 11, "bold"),
                relief="ridge",
                bd=2
            )
            server_frame.grid(row=0, column=col, padx=15, pady=15, sticky="n")

            for vm in server.vms:
                vm_frame = tk.LabelFrame(
                    server_frame,
                    text=f"VM {vm.id}",
                    padx=5,
                    pady=5,
                    bg="#b2f7b2",
                    relief="solid",
                    bd=1
                )
                vm_frame.pack(pady=5, fill="both")

                # عرض الـ Tasks الخاصة بهذه الـ VM
                for t in vm_tasks.get(vm.id, []):
                    tk.Label(
                        vm_frame,
                        text=f"Task {t}",
                        bg="#f7f7b2",
                        anchor="w"
                    ).pack(padx=3, pady=2, fill="x")

            col += 1

    # ------------------ Reset ------------------
    def reset_ui_DE(self):
        self.txt_de_output.delete("1.0", "end")
        self.progress_de["value"] = 0
        self.de_ax.clear()
        self.de_canvas.draw()
        self.de_summary_frame.destroy()
        self.de_summary_frame = tk.Frame(self.tab_de_results)
        self.de_summary_frame.pack(fill="x", pady=6)

    def build_compare_tab(self):
        # -------- Main container with vertical scroll ------------
        container = tk.Frame(self.tab_comparison)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame that will contain all widgets
        frm = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frm, anchor="nw")

        # تحديث الـ scroll region تلقائيًا عند تغير الحجم
        frm.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        tk.Label(frm, text="GA vs DE Comparison", font=("Arial", 16, "bold")).pack(pady=10)

        # -------- Top plot: GA + DE on same figure ------------
        top_frame = tk.Frame(frm)
        top_frame.pack(fill="both", expand=True)

        self.top_fig = Figure(figsize=(13.5, 6))
        self.top_ax = self.top_fig.add_subplot(111)
        self.top_canvas = FigureCanvasTkAgg(self.top_fig, master=top_frame)
        self.top_canvas.get_tk_widget().pack(fill="both", expand=True)

        # -------- Summary below top plot ------------
        summary_frame = tk.Frame(frm, bg="#f0f0f0", pady=5)
        summary_frame.pack(fill="x", pady=10)

        self.txt_ga_summary = tk.Text(summary_frame, height=20, width=60, bg="#e0f7ff", fg="blue",
                                      font=("Arial", 10, "bold"))
        self.txt_ga_summary.pack(side="left", padx=5, pady=2)
        self.txt_de_summary = tk.Text(summary_frame, height=20, width=60, bg="#fff4e0", fg="darkorange",
                                      font=("Arial", 10, "bold"))
        self.txt_de_summary.pack(side="right", padx=5, pady=2)



        # Initial update
        self.update_comparison()


    def update_comparison(self):
        import mplcursors
        import numpy as np

        # -------- تحقق من وجود بيانات --------
        if not hasattr(self, 'ga_all_runs') or not hasattr(self, 'de_all_runs'):
            return
        if not self.ga_all_runs or not self.de_all_runs:
            return

        # -------- أفضل Run للـ GA --------
        ga_min_values = [min(run) for run in self.ga_all_runs if run]
        if ga_min_values:
            self.best_ga_run_idx = int(np.argmin(ga_min_values))
            self.best_ga_history = self.ga_all_runs[self.best_ga_run_idx]
            self.best_ga_fit = min(self.best_ga_history)
            self.best_ga_gen = self.best_ga_history.index(self.best_ga_fit)
        else:
            self.best_ga_run_idx = 0
            self.best_ga_history = []
            self.best_ga_fit = 0
            self.best_ga_gen = 0

        # -------- أفضل Run للـ DE --------
        de_min_values = [min(run) for run in self.de_all_runs if run]
        if de_min_values:
            self.best_de_run_idx = int(np.argmin(de_min_values))
            self.best_de_history = self.de_all_runs[self.best_de_run_idx]
            self.best_de_fit = min(self.best_de_history)
            self.best_de_gen = self.best_de_history.index(self.best_de_fit) + 1
        else:
            self.best_de_run_idx = 0
            self.best_de_history = []
            self.best_de_fit = 0
            self.best_de_gen = 0

        # -------- Top plot GA vs DE (Best Runs) --------
        # self.top_ax.clear()
        # ga_line, = self.top_ax.plot(self.best_ga_history, label=f'GA Best Run {self.best_ga_run_idx + 1}', color='blue',
        #                             linestyle='-')
        # de_line, = self.top_ax.plot(self.best_de_history, label=f'DE Best Run {self.best_de_run_idx + 1}',
        #                             color='orange', linestyle='--')
        self.top_ax.clear()

        # محور X يبدأ من 1
        x_ga = range(1, len(self.best_ga_history) + 1)
        x_de = range(1, len(self.best_de_history) + 1)

        # رسم GA best run
        ga_line, = self.top_ax.plot(x_ga, self.best_ga_history,
                                    label=f'GA Best Run {self.best_ga_run_idx + 1}',
                                    color='blue', linestyle='-')

        # رسم DE best run
        de_line, = self.top_ax.plot(x_de, self.best_de_history,
                                    label=f'DE Best Run {self.best_de_run_idx + 1}',
                                    color='orange', linestyle='--')

        # Annotate best points
        self.top_ax.annotate(
            f"{self.best_ga_fit:.4f}\nGen:{self.best_ga_gen+1}",
            xy=(self.best_ga_gen, self.best_ga_fit),
            xytext=(self.best_ga_gen, self.best_ga_fit + 0.5),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            ha='center', fontsize=10, color='blue', fontweight='bold'
        )
        self.top_ax.annotate(
            f"{self.best_de_fit:.4f}\nGen:{self.best_de_gen}",
            xy=(self.best_de_gen - 1, self.best_de_fit),
            xytext=(self.best_de_gen - 1, self.best_de_fit + 0.5),
            arrowprops=dict(facecolor='orange', shrink=0.05),
            ha='center', fontsize=10, color='darkorange', fontweight='bold'
        )

        self.top_ax.set_title("GA vs DE Best Run Comparison", fontsize=14, fontweight='bold')
        self.top_ax.set_xlabel("Generation", fontsize=12)
        self.top_ax.set_ylabel("Fitness", fontsize=12)
        self.top_ax.grid(True, linestyle='--', alpha=0.6)
        self.top_ax.legend()
        self.top_canvas.draw()

        # -------- Hover على أفضل Run --------
        cursor = mplcursors.cursor([ga_line, de_line], hover=True)

        @cursor.connect("add")
        def on_add(sel):
            gen = int(sel.target[0]) + 1
            val = sel.target[1]
            algo = "GA" if sel.artist == ga_line else "DE"
            run_number = self.best_ga_run_idx + 1 if algo == "GA" else self.best_de_run_idx + 1
            sel.annotation.set_text(f"{algo} Best Run {run_number}\nGen:{gen}\nFitness:{val:.4f}")
            sel.annotation.get_bbox_patch().set_facecolor(sel.artist.get_color())
            sel.annotation.get_bbox_patch().set_alpha(0.8)

        # -------- Update GA summary --------
        self.txt_ga_summary.delete("1.0", "end")
        self.txt_ga_summary.configure(font=("Arial", 13, "bold"), fg="blue")
        self.txt_ga_summary.insert("end", f"Algorithm: GA\n\n")
        self.txt_ga_summary.insert("end", f"Best Run: {self.best_ga_run_idx + 1}\n")
        self.txt_ga_summary.insert("end", f"Found at Generation: {self.best_ga_gen+1}\n")
        self.txt_ga_summary.insert("end", f"Total Generations: {len(self.ga_history)}\n")
        self.txt_ga_summary.insert("end", f"Best Fitness: {self.best_ga_fit:.4f}\n")




        if hasattr(self, 'ga_all_runs'):
            avg_fit = sum([min(run) for run in self.ga_all_runs]) / len(self.ga_all_runs)
            self.txt_ga_summary.insert("end", f"Average Best Fitness over Runs: {avg_fit:.4f}\n\n")
        self.txt_ga_summary.insert("end", f"Time to Best: {getattr(self, 'ga_first_best_time', 0):.2f}s\n")

        # -------- Update DE summary --------
        self.txt_de_summary.delete("1.0", "end")
        self.txt_de_summary.configure(font=("Arial", 13, "bold"), fg="darkorange")
        self.txt_de_summary.insert("end", f"Algorithm: DE\n\n")
        self.txt_de_summary.insert("end", f"Best Run: {self.best_de_run_idx + 1}\n")
        self.txt_de_summary.insert("end", f"Found at Generation: {self.best_de_gen}\n")
        self.txt_de_summary.insert("end", f"Total Generations: {len(self.de_history)}\n")
        self.txt_de_summary.insert("end", f"Best Fitness: {self.best_de_fit:.4f}\n")


        if hasattr(self, 'de_all_runs'):
            avg_fit = sum([min(run) for run in self.de_all_runs]) / len(self.de_all_runs)
            self.txt_de_summary.insert("end", f"Average Best Fitness over Runs: {avg_fit:.4f}\n\n")
        self.txt_de_summary.insert("end", f"Time to Best: {getattr(self, 'de_first_best_time', 0):.2f}s\n")



# ------------------- Run App -------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()








