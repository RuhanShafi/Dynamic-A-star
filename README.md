# Dynamic & Interactive A* Visualizer


A simple A* Visualizer written in Python where the user can define a start & end point and create obstacles and see a visualization of the A* algorithm trying to find the optimal path from start to finish. I built this project to learn how to implement A* algorithm in a real world scenario and to learn the basics of building GUIs in the Python `tkinter` library.
 
## Technologies Used - Tech Stack
- `python`
- `numpy`
- `tkinter`
## Features

### Front end
- Grid Creation: By default, the program will create a `20x20` grid, however, the user has the ability to define the dimensions of the grid to their requirements and the program will create it for you, (as long as it's a positive number, don't worry there are safeguards to obey the rules of geometry). All the user need to do is use the two input boxes to define their dimensions.
- Visualizer Speed Slider - Control the speed of the A* visualization with the slider to your liking - This has no affect on the actual computational speed of the algorithm, this only affects the visualization that the user sees.

### Back end - Optimization strategies and implementation
#### Memory Management
- Binary heap (heapq): O(log n) insertions instead of O(n) for arrays
- Hash set for closed list: O(1) lookups instead of O(n) list searches

## Areas for Improvements / TODO list
- UI Overhaul - Front end
- 
