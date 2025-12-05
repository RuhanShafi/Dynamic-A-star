import tkinter as tk
from tkinter import BOTH, ttk
from tkinter import messagebox
from typing import Self

# from enum import Enum
# from typing import Optional, List, Tuple
import numpy as np

# from math import sqrt
import heapq
# import matplotlib as plt


class AStarAlgorithm:
    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.rows, self.cols = grid.shape

    def heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        """Get valid neighboring cells (not walls, within bounds)"""
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0:
                neighbors.append((r, c))
        return neighbors

    def find_path(self):
        """
        Execute A* algorithm
        Returns: (visited_nodes, path)
            visited_nodes: list of nodes explored in order
            path: list of nodes in the final path (empty if no path found)
        """
        if not self.start or not self.end:
            return [], []

        frontier = []
        heapq.heappush(frontier, (0, self.start))
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        visited_order = []

        while frontier:
            current = heapq.heappop(frontier)[1]
            visited_order.append(current)

            if current == self.end:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return visited_order, path

            for next_node in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, self.end)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        # No path found
        return visited_order, []


class UI:
    def __init__(self, root):
        self.root = root
        self.root.title = "A* Testbed"

        # Grid Variables
        self.cols = 10
        self.rows = 10
        self.grid = None

        self.create_weigets()

    def create_weigets(self):
        control_panel = ttk.Frame(self.root, padding="10")
        control_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            control_panel,
            text="A*",
            font=("Arial", 24, "bold"),
            bg="#1e1e2e",
            fg="white",
        )
        title_label.pack(pady=(0, 5))

        title_sub = tk.Label(
            control_panel,
            text="Placeholder",
            font=("Arial", 11),
            bg="#1e1e2e",
            fg="white",
        )
        title_sub.pack(pady=(0, 20))

        sidebar_frame = tk.Frame(control_panel, bg="#1e1e2e")
        sidebar_frame.pack(fill=tk.BOTH, expand=True)

    def create_grid(self):
        try:
            if self.cols <= 0 or self.rows <= 0:
                raise ValueError("Dimesions must be positive")
            # Creating the Numpy Grid used in backend
            self.grid = np.zeros((self.rows, self.cols))
        except ValueError as e:
            print(f"Invalid Input: {e}")

    def clear_grid(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = UI(root)
    root.mainloop()
