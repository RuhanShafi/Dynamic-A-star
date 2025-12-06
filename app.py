import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import heapq
from collections import deque


class Node:
    """Lightweight node class for A* algorithm"""

    __slots__ = ("pos", "g", "h", "f", "parent")

    def __init__(self, pos, g=0, h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f


class AStarAlgorithm:
    """Optimized A* pathfinding algorithm"""

    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.rows, self.cols = grid.shape

    def heuristic(self, pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def get_neighbors(self, pos):
        r, c = pos
        neighbors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self):
        if not self.start or not self.end:
            return [], []

        open_heap = [(0, 0, self.start)]
        closed_set = set()
        g_scores = {self.start: 0}
        parents = {self.start: None}
        visited_order = []
        counter = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)
            visited_order.append(current)

            if current == self.end:
                path = self._reconstruct_path(parents)
                return visited_order, path

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h = self.heuristic(neighbor, self.end)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(open_heap, (f, h, neighbor))
                    parents[neighbor] = current

        return visited_order, []

    def _reconstruct_path(self, parents):
        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = parents[current]
        return list(reversed(path))


class PathfindingVisualizerUI:
    """Clean UI matching the screenshot design"""

    def __init__(self, root):
        self.root = root
        self.root.title("A* Pathfinding Visualizer")
        self.root.geometry("1200x800")

        # State variables
        self.grid = None
        self.rows = 20
        self.cols = 20
        self.mode = "wall"
        self.start = None
        self.end = None
        self.animation_speed = 50
        self.animating = False
        self.mouse_pressed = False
        self.last_cell = None

        self.create_ui()
        self.create_grid()

    def create_ui(self):
        # Top control panel
        control_frame = tk.Frame(self.root, bg="#e5e5e5", relief=tk.FLAT)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Row 1: Grid dimensions
        row1 = tk.Frame(control_frame, bg="#e5e5e5")
        row1.pack(anchor=tk.W, pady=5)

        tk.Label(row1, text="Rows:", bg="#e5e5e5").pack(side=tk.LEFT, padx=5)
        self.rows_entry = tk.Entry(row1, width=8)
        self.rows_entry.insert(0, "20")
        self.rows_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row1, text="Cols:", bg="#e5e5e5").pack(side=tk.LEFT, padx=15)
        self.cols_entry = tk.Entry(row1, width=8)
        self.cols_entry.insert(0, "20")
        self.cols_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(
            row1,
            text="Create Grid",
            command=self.create_grid,
            relief=tk.RAISED,
            padx=10,
            pady=2,
        ).pack(side=tk.LEFT, padx=15)

        # Row 2: Mode selection and actions
        row2 = tk.Frame(control_frame, bg="#e5e5e5")
        row2.pack(anchor=tk.W, pady=5)

        tk.Label(row2, text="Mode:", bg="#e5e5e5").pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="wall")

        tk.Radiobutton(
            row2,
            text="Wall",
            variable=self.mode_var,
            value="wall",
            bg="#e5e5e5",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            row2,
            text="Start",
            variable=self.mode_var,
            value="start",
            bg="#e5e5e5",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            row2,
            text="End",
            variable=self.mode_var,
            value="end",
            bg="#e5e5e5",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            row2,
            text="Find Path (A*)",
            command=self.find_path,
            relief=tk.RAISED,
            padx=10,
            pady=2,
        ).pack(side=tk.LEFT, padx=15)
        tk.Button(
            row2,
            text="Clear Path",
            command=self.clear_path,
            relief=tk.RAISED,
            padx=10,
            pady=2,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            row2,
            text="Clear All",
            command=self.clear_all,
            relief=tk.RAISED,
            padx=10,
            pady=2,
        ).pack(side=tk.LEFT, padx=5)

        # Row 3: Speed control
        row3 = tk.Frame(control_frame, bg="#e5e5e5")
        row3.pack(anchor=tk.W, pady=5)

        tk.Label(row3, text="Speed:", bg="#e5e5e5").pack(side=tk.LEFT, padx=5)

        self.speed_scale = tk.Scale(
            row3,
            from_=1,
            to=200,
            orient=tk.HORIZONTAL,
            command=self.update_speed,
            length=200,
            bg="#e5e5e5",
            relief=tk.FLAT,
            highlightthickness=0,
        )
        self.speed_scale.set(50)
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Canvas area
        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def change_mode(self):
        self.mode = self.mode_var.get()

    def update_speed(self, value):
        self.animation_speed = int(201 - float(value))

    def create_grid(self):
        try:
            self.rows = int(self.rows_entry.get())
            self.cols = int(self.cols_entry.get())

            if self.rows <= 0 or self.cols <= 0:
                raise ValueError("Dimensions must be positive")

            self.grid = np.zeros((self.rows, self.cols))
            self.start = None
            self.end = None
            self.animating = False
            self.mouse_pressed = False
            self.last_cell = None

            self.update_display()

        except ValueError as e:
            print(f"Invalid input: {e}")

    def clear_path(self):
        if self.grid is not None:
            self.animating = False
            self.update_display(full_redraw=True)

    def clear_all(self):
        if self.grid is not None:
            self.grid = np.zeros((self.rows, self.cols))
            self.start = None
            self.end = None
            self.animating = False
            self.update_display(full_redraw=True)

    def update_display(self, full_redraw=True):
        if full_redraw:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            self.fig = Figure(figsize=(10, 8), facecolor="white")
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor("white")

            display_grid = self.create_display_grid()
            self.im = self.ax.imshow(
                display_grid, interpolation="nearest", origin="upper"
            )

            self.draw_points()

            self.ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
            self.ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
            self.ax.tick_params(which="minor", size=0)

            self.ax.set_xticks(np.arange(0, self.cols, 5))
            self.ax.set_yticks(np.arange(0, self.rows, 5))

            self.ax.set_title(f"A* Pathfinding Visualizer ({self.rows}x{self.cols})")

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
            self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
            self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        else:
            display_grid = self.create_display_grid()
            self.im.set_data(display_grid)
            self.draw_points()
            self.canvas.draw_idle()

    def create_display_grid(self):
        display_grid = np.ones((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 1:
                    display_grid[i, j] = [0, 0, 0]  # Black walls
        return display_grid

    def draw_points(self):
        # Clear old patches
        for patch in self.ax.patches[:]:
            patch.remove()

        # Draw start (green)
        if self.start:
            patch = Rectangle(
                (self.start[1] - 0.4, self.start[0] - 0.4),
                0.8,
                0.8,
                facecolor="green",
                edgecolor="darkgreen",
                linewidth=2,
            )
            self.ax.add_patch(patch)

        # Draw end (red)
        if self.end:
            patch = Rectangle(
                (self.end[1] - 0.4, self.end[0] - 0.4),
                0.8,
                0.8,
                facecolor="red",
                edgecolor="darkred",
                linewidth=2,
            )
            self.ax.add_patch(patch)

    def on_mouse_press(self, event):
        if event.inaxes != self.ax or self.animating:
            return
        self.mouse_pressed = True
        self.process_cell(event)

    def on_mouse_release(self, event):
        self.mouse_pressed = False
        self.last_cell = None

    def on_mouse_motion(self, event):
        if not self.mouse_pressed or event.inaxes != self.ax or self.animating:
            return
        self.process_cell(event)

    def process_cell(self, event):
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        if 0 <= row < self.rows and 0 <= col < self.cols:
            if (row, col) == self.last_cell and self.mode == "wall":
                return

            self.last_cell = (row, col)

            if self.mode == "wall":
                if self.mouse_pressed and event.name == "motion_notify_event":
                    self.grid[row, col] = 1
                else:
                    self.grid[row, col] = 1 - self.grid[row, col]
            elif self.mode == "start":
                self.start = (row, col)
                self.grid[row, col] = 0
            elif self.mode == "end":
                self.end = (row, col)
                self.grid[row, col] = 0

            self.update_display(full_redraw=False)

    def find_path(self):
        if not self.start or not self.end or self.animating:
            return

        self.animating = True
        self.update_display(full_redraw=True)

        algorithm = OptimizedAStarAlgorithm(self.grid, self.start, self.end)
        visited_order, path = algorithm.find_path()

        self.animate_search(visited_order, path)

    def animate_search(self, visited, path):
        self.visited_queue = deque(visited)
        self.path_queue = deque(path)
        self.visited_patches = []
        self.path_patches = []
        self.animate_step()

    def animate_step(self):
        if self.visited_queue:
            node = self.visited_queue.popleft()
            if node != self.start and node != self.end:
                patch = Rectangle(
                    (node[1] - 0.4, node[0] - 0.4),
                    0.8,
                    0.8,
                    facecolor="lightblue",
                    edgecolor="blue",
                    linewidth=0.5,
                )
                self.ax.add_patch(patch)
                self.visited_patches.append(patch)
            self.canvas.draw()
            self.root.after(self.animation_speed, self.animate_step)
        elif self.path_queue:
            # Clear visited patches when starting to draw path
            if self.path_queue and not self.path_patches:
                for patch in self.visited_patches:
                    patch.remove()
                self.visited_patches.clear()
                self.canvas.draw()

            node = self.path_queue.popleft()
            if node != self.start and node != self.end:
                patch = Rectangle(
                    (node[1] - 0.4, node[0] - 0.4),
                    0.8,
                    0.8,
                    facecolor="yellow",
                    edgecolor="orange",
                    linewidth=2,
                )
                self.ax.add_patch(patch)
                self.path_patches.append(patch)
            self.canvas.draw()
            self.root.after(self.animation_speed, self.animate_step)
        else:
            self.animating = False


if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingVisualizerUI(root)
    root.mainloop()
