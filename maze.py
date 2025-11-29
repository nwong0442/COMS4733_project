import pygame
import random
from collections import deque
import heapq
import sys # Import sys for clean exit

# --- 1. CONFIGURATION ---
WIDTH = 600 # 30 cells * 20 pixels/cell
HEIGHT = 600 # 30 cells * 20 pixels/cell
CELL_SIZE = 20 
COLS = 30 # WIDTH // CELL_SIZE (600/20)
ROWS = 30 # HEIGHT // CELL_SIZE (600/20)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0) # Start
RED = (255, 0, 0)   # End
YELLOW = (255, 255, 0) # Final Path
LIGHT_BLUE = (173, 216, 230) # Explored Nodes

# Start and End points (row, col coordinates)
START = (ROWS - 1, 0) 
END = (0, COLS - 1) 

# Central Obstacle Coordinates (Rows 10-19, Cols 10-19)
OBS_R_START = 10
OBS_R_END = 20
OBS_C_START = 10
OBS_C_END = 20

# Global list to store removed walls (Wall ID: ((r, c), direction))
REMOVED_WALLS = []


# --- 2. CELL CLASS ---
class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y # x=col, y=row
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False
        self.parent = None 
        self.distance = float('inf') 

    # Needed for the tie-breaker fix: Python needs to be able to compare Cell objects
    def __lt__(self, other):
        return (self.y, self.x) < (other.y, other.x)

    def draw(self, screen):
        x, y = self.x * CELL_SIZE, self.y * CELL_SIZE
        
        is_obstacle = (OBS_R_START <= self.y < OBS_R_END and 
                       OBS_C_START <= self.x < OBS_C_END)
        
        if is_obstacle:
            # Draw the obstacle area as a solid BLACK block
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect)
            return

        # Draw the walls
        if self.walls['N']:
            pygame.draw.line(screen, BLACK, (x, y), (x + CELL_SIZE, y), 2)
        if self.walls['E']:
            pygame.draw.line(screen, BLACK, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls['S']:
            pygame.draw.line(screen, BLACK, (x + CELL_SIZE, y + CELL_SIZE), (x, y + CELL_SIZE), 2)
        if self.walls['W']:
            pygame.draw.line(screen, BLACK, (x, y + CELL_SIZE), (x, y), 2)

    def draw_path(self, screen, color):
        x, y = self.x * CELL_SIZE, self.y * CELL_SIZE
        rect = pygame.Rect(x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, color, rect)

# --- 3. MAZE GENERATION UTILITIES ---
def is_in_obstacle(r, c):
    """Checks if a cell (row r, col c) is inside the central 10x10 obstacle."""
    return (OBS_R_START <= r < OBS_R_END and 
            OBS_C_START <= c < OBS_C_END)

def check_cell(x, y, grid):
    """Returns the cell at (x, y) if it's in bounds AND not in the obstacle, otherwise returns None."""
    if 0 <= x < COLS and 0 <= y < ROWS:
        if is_in_obstacle(y, x):
            return None
        return grid[y][x]
    return None

def remove_wall(current, next_cell):
    """
    Removes the wall between two adjacent cells and records the removed wall.
    The wall is recorded using the coordinates of the 'current' cell.
    """
    global REMOVED_WALLS
    dx = current.x - next_cell.x
    dy = current.y - next_cell.y
    
    wall_key = None

    if dx == 1:  # next_cell is West of current
        current.walls['W'] = False
        next_cell.walls['E'] = False
        wall_key = 'W'
    elif dx == -1: # next_cell is East of current
        current.walls['E'] = False
        next_cell.walls['W'] = False
        wall_key = 'E'
    
    if dy == 1:  # next_cell is North of current
        current.walls['N'] = False
        next_cell.walls['S'] = False
        wall_key = 'N'
    elif dy == -1: # next_cell is South of current
        current.walls['S'] = False
        next_cell.walls['N'] = False
        wall_key = 'S'

    # Store the unique identifier for the removed wall: ((row, col), direction)
    if wall_key:
        # We only record the wall break from the perspective of the *current* cell.
        # The reconstruction logic handles the neighboring wall.
        removed_wall_id = ((current.y, current.x), wall_key)
        REMOVED_WALLS.append(removed_wall_id)


def generate_maze(start_cell, grid):
    """Generates the maze using Recursive Backtracking (DFS). Skips obstacle area."""
    # Reset walls list before generating a new maze
    global REMOVED_WALLS
    REMOVED_WALLS = [] 
    
    random.seed(2) # Keep the seed for consistent maze

    stack = [start_cell]
    start_cell.visited = True

    while stack:
        current_cell = stack[-1]
        
        neighbors = []
        directions = [
            (0, -1, 'N'), # North (dx, dy, wall)
            (0, 1, 'S'),  # South
            (1, 0, 'E'),  # East
            (-1, 0, 'W')  # West
        ]
        
        for dx, dy, _ in directions:
            next_cell = check_cell(current_cell.x + dx, current_cell.y + dy, grid)
            if next_cell and not next_cell.visited:
                neighbors.append(next_cell)

        if neighbors:
            next_cell = random.choice(neighbors)
            remove_wall(current_cell, next_cell)
            next_cell.visited = True
            stack.append(next_cell)
        else:
            stack.pop()

def reconstruct_maze_from_walls(removed_walls):
    """
    Reconstructs the maze grid structure based on a list of removed wall IDs.
    All walls are initialized as present (True).
    
    NOTE: The main execution flow *does not* use this reconstructed grid for 
    solving, but it's the function the prompt is specifically interested in 
    as the 'reconstruction' logic.
    """
    # 1. Initialize a new grid with all walls present
    new_grid = [[Cell(c, r) for c in range(COLS)] for r in range(ROWS)]
    
    # 2. Apply the wall removal instructions
    for wall_id in removed_walls:
        # wall_id format: ((r, c), direction)
        (r, c), direction = wall_id
        
        # Get the cell where the wall removal is defined
        current_cell = new_grid[r][c]
        
        # 3. Determine the neighboring cell and its corresponding wall key
        dr, dc, opposite_direction = 0, 0, ''
        
        if direction == 'N':
            dr, dc, opposite_direction = -1, 0, 'S'
        elif direction == 'S':
            dr, dc, opposite_direction = 1, 0, 'N'
        elif direction == 'W':
            dr, dc, opposite_direction = 0, -1, 'E'
        elif direction == 'E':
            dr, dc, opposite_direction = 0, 1, 'W'
            
        nr, nc = r + dr, c + dc
        
        # Check if the neighbor exists in the *new* grid (boundary check)
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            next_cell = new_grid[nr][nc]
            
            # Remove the walls in both cells
            current_cell.walls[direction] = False
            next_cell.walls[opposite_direction] = False
            
    return new_grid

def create_n_shape_opening(grid):
    """Placeholder: This function is not used for the central obstacle design."""
    pass


# --- 4. DIJKSTRA'S ALGORITHM IMPLEMENTATION ---
def get_neighbors(cell, grid):
    """Returns a list of accessible (open-path) neighbors for a cell."""
    neighbors = []
    r, c = cell.y, cell.x # row, col

    # Directions: (dr, dc, wall_key)
    moves = [
        (-1, 0, 'N'), (1, 0, 'S'),  
        (0, 1, 'E'), (0, -1, 'W') 
    ]

    for dr, dc, wall_key in moves:
        nr, nc = r + dr, c + dc
        # Check for cell existence (bounds & obstacle) and open wall
        next_cell = check_cell(nc, nr, grid) 
        
        if next_cell and not cell.walls[wall_key]:
            neighbors.append(next_cell)
            
    return neighbors

def dijkstra_search(grid, start_pos, end_pos):
    """Implements Dijkstra's algorithm to find the shortest path."""
    if is_in_obstacle(start_pos[0], start_pos[1]) or is_in_obstacle(end_pos[0], end_pos[1]):
         print("Error: Start or End position is inside the obstacle.")
         return {}, []
         
    start_cell = grid[start_pos[0]][start_pos[1]]
    end_cell = grid[end_pos[0]][end_pos[1]]
    
    tie_breaker = 0 
    pq = [(0, tie_breaker, start_cell)]
    start_cell.distance = 0
    visited_nodes = {}
    expansion_order = []
    
    while pq:
        current_distance, _, current_cell = heapq.heappop(pq)
        
        if current_distance > current_cell.distance:
            continue
        
        expansion_order.append(current_cell)
        
        if current_cell == end_cell:
            break
        
        visited_nodes[current_cell] = True

        for neighbor in get_neighbors(current_cell, grid):
            new_distance = current_distance + 1 
            
            if new_distance < neighbor.distance:
                neighbor.distance = new_distance
                neighbor.parent = current_cell
                
                tie_breaker += 1
                heapq.heappush(pq, (new_distance, tie_breaker, neighbor))
                
    return visited_nodes, expansion_order


def reconstruct_path(end_cell):
    """Reconstructs the path from the end_cell back to the start using parent pointers."""
    path = deque()
    current = end_cell
    while current and current.parent: 
        path.appendleft(current)
        current = current.parent
    if current:
        path.appendleft(current) 
    return path

# --- 5. CORE MAZE FUNCTION ---
def generate_and_solve_maze(start=START, end=END):
    """
    Generates the maze, finds the shortest path, and returns the visualization 
    data AND the list of removed walls.
    """
    # 1. Create the grid 
    grid = [[Cell(c, r) for c in range(COLS)] for r in range(ROWS)]
    
    # 2. Generate the maze (REMOVED_WALLS list is populated here)
    start_cell_gen = grid[0][0] 
    generate_maze(start_cell_gen, grid)
    
    # Store the list of removed walls *after* generation is complete
    final_removed_walls = REMOVED_WALLS 

    # 3. Reset for search
    for r in range(ROWS):
        for c in range(COLS):
            if not is_in_obstacle(r, c):
                grid[r][c].visited = False
                grid[r][c].distance = float('inf')
                grid[r][c].parent = None

    # 4. Run Dijkstra's Algorithm
    visited_nodes, expansion_order = dijkstra_search(grid, start, end)
    
    # 5. Reconstruct the path 
    end_cell = grid[end[0]][end[1]]
    path = reconstruct_path(end_cell)

    # 6. Convert path to instructions
    move_instructions = generate_move_instructions(path)

    print(f"--- Shortest Path Instructions (Start: ({START[0]}, {START[1]}), End: ({END[0]}, {END[1]})) ---")
    print(move_instructions)
    print(f"Total steps: {len(move_instructions)}")
    print(f"\nTotal Removed Walls: {len(final_removed_walls)}")
    
    # Return all necessary data
    return grid, path, visited_nodes, expansion_order, final_removed_walls


def generate_move_instructions(path):
    instructions = []
    for i in range(len(path) - 1):
        current_cell = path[i]
        next_cell = path[i+1]
        d_row = next_cell.y - current_cell.y
        d_col = next_cell.x - current_cell.x
        
        if d_row == -1 and d_col == 0:
            move = 'U' # Up
        elif d_row == 1 and d_col == 0:
            move = 'D' # Down
        elif d_col == -1 and d_row == 0:
            move = 'L' # Left
        elif d_col == 1 and d_row == 0:
            move = 'R' # Right
        else: move = None
        
        if move: instructions.append(move)
    return instructions


# --- 6. VISUALIZATION FUNCTION ---
def visualize_maze(grid, path, visited_nodes, expansion_order):
    """Initializes Pygame and runs the visualization loop."""
    try:
        pygame.init()
    except pygame.error as e:
        print(f"Pygame initialization failed: {e}")
        print("Please ensure you have a display server running if accessing remotely.")
        return
        
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("30x30 Maze with Central Obstacle")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(WHITE)

        # Draw explored nodes
        for cell in visited_nodes:
            # We skip drawing a path marker for the start cell itself since it will be drawn over by GREEN
            if (cell.y, cell.x) != START and (cell.y, cell.x) != END:
                cell.draw_path(screen, LIGHT_BLUE) 
        
        # Draw final path
        for cell in path:
            cell.draw_path(screen, YELLOW)
            
        # Draw walls and obstacles
        for r in range(ROWS):
            for c in range(COLS):
                grid[r][c].draw(screen)
        
        # Draw Start/End points last to ensure visibility
        grid[START[0]][START[1]].draw_path(screen, GREEN) # Start
        grid[END[0]][END[1]].draw_path(screen, RED)       # End

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


# --- 7. HELPER FUNCTIONS ---
def convert_david_to_mujoco_format(removed_walls, grid_size):
    """
    Converts wall format ((row, col), direction) to 
    MuJoCo cell pair format (cell1, cell2).
    
    Args:
        removed_walls: List of ((row, col), direction) tuples
        grid_size: Number of cells per side (e.g., 30 for this maze)
    
    Returns:
        List of (cell1, cell2) tuples where cells are numbered 1 to grid_size²
    """
    cell_pairs = []
    
    for (row, col), direction in removed_walls:
        # Convert (row, col) to cell number (1-indexed)
        cell1 = row * grid_size + col + 1
        
        # Find the neighboring cell based on direction
        if direction == 'N':
            cell2 = cell1 - grid_size  # Cell above
        elif direction == 'S':
            cell2 = cell1 + grid_size  # Cell below
        elif direction == 'W':
            cell2 = cell1 - 1  # Cell to the left
        elif direction == 'E':
            cell2 = cell1 + 1  # Cell to the right
        else:
            continue
        
        # Only add if cell2 is valid
        if 1 <= cell2 <= grid_size * grid_size:
            cell_pairs.append((cell1, cell2))
    
    return cell_pairs


def get_removed_walls():
    """
    Returns the list of removed walls in MuJoCo format (cell1, cell2).
    Call this AFTER generate_and_solve_maze() has been run.
    
    NOTE: This generates a new maze each time it's called. If you need
    the same maze across multiple calls, call generate_and_solve_maze()
    once and store the result.
    """
    _, _, _, _, removed_walls = generate_and_solve_maze()
    return convert_david_to_mujoco_format(removed_walls, COLS)


def start_end_to_instruction():
    """
    Generates the maze and returns the path instructions.
    This is the function that project.ipynb calls!
    
    Returns:
        List of move instructions ('U', 'D', 'L', 'R')
    """
    # Generate maze and solve it (without visualization)
    grid, path, visited_nodes, expansion_order, removed_walls = generate_and_solve_maze()
    
    # Return the move instructions
    instructions = generate_move_instructions(path)
    return instructions


# --- 8. MAIN EXECUTION ---
def main():
    # 1. Generate the maze and solve it, getting visualization data and removed walls list
    grid, path, visited_nodes, expansion_order, removed_walls = generate_and_solve_maze()
    
    # print("\n--- List of Removed Walls (ID: ((Row, Col), Direction)) ---")
    
    # You can call the reconstruction function here if you needed to test it
    # grid = reconstruct_maze_from_walls(removed_walls) 
    
    # 3. Call the visualization function
    # NOTE: This requires a graphical environment (Pygame) to run successfully.
    visualize_maze(grid, path, visited_nodes, expansion_order)


if __name__ == "__main__":
    main()