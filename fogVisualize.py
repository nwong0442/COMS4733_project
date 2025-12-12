import pygame
import random
from collections import deque
import heapq
import sys
import math

# --- 1. CONFIGURATION ---
WIDTH = 600
HEIGHT = 600
CELL_SIZE = 20
COLS = 30
ROWS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)   # Robot (Start)
RED = (255, 0, 0)     # Goal (End)
YELLOW = (255, 255, 0) # Current Planned Path
LIGHT_BLUE = (173, 216, 230) # Robot History/Trail
GRAY = (100, 100, 100) # Fog of War (unexplored)

# Start and End points (row, col coordinates)
START_POS = (ROWS - 1, 0) # Bottom-left
END_POS = (0, COLS - 1)   # Top-right

# Central Obstacle Coordinates
OBS_R_START = 10
OBS_R_END = 20
OBS_C_START = 10
OBS_C_END = 20

# Global list to store removed walls
REMOVED_WALLS = []

# Simulation Globals
FOG_OF_WAR_N = 3 #Can edit 3,5,7,9,11...
GLOBAL_MAZE_GRID = None 
FOG_GRID = None         

# --- 2. CELL CLASS ---
class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False 
        self.parent = None 
        
        self.g = float('inf') 
        self.f = float('inf')

        self.known = False      

    def __lt__(self, other):
        return self.f < other.f

    def draw(self, screen, is_fog_of_war_draw=False, current_robot_pos=None):
        x, y = self.x * CELL_SIZE, self.y * CELL_SIZE
        
        is_obstacle = (OBS_R_START <= self.y < OBS_R_END and 
                       OBS_C_START <= self.x < OBS_C_END)
        
        if is_obstacle:
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect)
            return

        if is_fog_of_war_draw and not self.known:
             rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
             pygame.draw.rect(screen, GRAY, rect)
             return

        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, WHITE, rect)

        wall_color = BLACK
        if self.walls['N']:
            pygame.draw.line(screen, wall_color, (x, y), (x + CELL_SIZE, y), 2)
        if self.walls['E']:
            pygame.draw.line(screen, wall_color, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
        if self.walls['S']:
            pygame.draw.line(screen, wall_color, (x + CELL_SIZE, y + CELL_SIZE), (x, y + CELL_SIZE), 2)
        if self.walls['W']:
            pygame.draw.line(screen, wall_color, (x, y + CELL_SIZE), (x, y), 2)

    def draw_path(self, screen, color):
        x, y = self.x * CELL_SIZE, self.y * CELL_SIZE
        rect = pygame.Rect(x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, color, rect)

# --- 3. MAZE GENERATION ---
def is_in_obstacle(r, c):
    return (OBS_R_START <= r < OBS_R_END and OBS_C_START <= c < OBS_C_END)

def check_cell(x, y, grid):
    if 0 <= x < COLS and 0 <= y < ROWS:
        if is_in_obstacle(y, x):
            return None
        return grid[y][x]
    return None

def remove_wall(current, next_cell):
    """
    Removes wall between current and next_cell and records the action.
    """
    dx = current.x - next_cell.x
    dy = current.y - next_cell.y
    
    wall_removed = None

    if dx == 1:  
        current.walls['W'] = False
        next_cell.walls['E'] = False
        wall_removed = 'W'
    elif dx == -1: 
        current.walls['E'] = False
        next_cell.walls['W'] = False
        wall_removed = 'E'
    elif dy == 1:  
        current.walls['N'] = False
        next_cell.walls['S'] = False
        wall_removed = 'N'
    elif dy == -1: 
        current.walls['S'] = False
        next_cell.walls['N'] = False
        wall_removed = 'S'

    if wall_removed:
        REMOVED_WALLS.append(((current.y, current.x), wall_removed))

def generate_maze(start_cell, grid):
    global REMOVED_WALLS
    REMOVED_WALLS = [] # Reset list

    random.seed(2)
    stack = [start_cell]
    start_cell.visited = True
    while stack:
        current_cell = stack[-1]
        neighbors = []
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        for dx, dy in directions:
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

# --- 4. SENSING AND A* SOLVER ---

def manhattan_distance(cell1, cell2):
    return abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)

def sense_area(robot_pos, global_grid, fog_grid, n=FOG_OF_WAR_N):
    """Reveals the map around the robot."""
    r_center, c_center = robot_pos
    view_radius = math.ceil(n / 2)
    
    r_start = max(0, r_center - view_radius + 1)
    r_end = min(ROWS, r_center + view_radius)
    c_start = max(0, c_center - view_radius + 1)
    c_end = min(COLS, c_center + view_radius)
    
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if not is_in_obstacle(r, c):
                fog_cell = fog_grid[r][c]
                global_cell = global_grid[r][c]
                
                if not fog_cell.known:
                    fog_cell.known = True
                
                fog_cell.walls = global_cell.walls.copy()

def get_neighbors_astar(cell, grid):
    """Returns valid neighbors based on CURRENT KNOWN walls."""
    neighbors = []
    r, c = cell.y, cell.x
    moves = [(-1, 0, 'N'), (1, 0, 'S'), (0, 1, 'E'), (0, -1, 'W')] 

    for dr, dc, wall_key in moves:
        if cell.walls[wall_key]: 
            continue
            
        nr, nc = r + dr, c + dc
        next_cell = check_cell(nc, nr, grid)
        
        if next_cell:
            opp_wall_key = {'N':'S', 'S':'N', 'E':'W', 'W':'E'}[wall_key]
            if not next_cell.walls[opp_wall_key]:
                neighbors.append(next_cell)
            
    return neighbors

def run_a_star(start_node, end_node, grid):
    """Runs A* Search on the Fog Grid."""
    for r in range(ROWS):
        for c in range(COLS):
            grid[r][c].g = float('inf')
            grid[r][c].f = float('inf')
            grid[r][c].parent = None

    open_set = []
    
    start_node.g = 0
    start_node.f = manhattan_distance(start_node, end_node)
    heapq.heappush(open_set, (start_node.f, start_node))
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == end_node:
            return True 

        for neighbor in get_neighbors_astar(current, grid):
            temp_g = current.g + 1 
            
            if temp_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = temp_g
                neighbor.f = temp_g + manhattan_distance(neighbor, end_node)
                
                if (neighbor.f, neighbor) not in open_set:
                     heapq.heappush(open_set, (neighbor.f, neighbor))
                     
    return False

def reconstruct_path(end_node):
    path = []
    current = end_node
    while current.parent:
        path.append(current)
        current = current.parent
    return path[::-1]

def generate_and_solve_maze_2(start_pos):
    """Main Loop: Sense -> Update -> Re-plan (Repeated A*)."""
    setup()
    # SENSE
    sense_area(start_pos, GLOBAL_MAZE_GRID, FOG_GRID)
    
    # PLAN (Repeated A*)
    start_cell = FOG_GRID[start_pos[0]][start_pos[1]]
    end_cell = FOG_GRID[END_POS[0]][END_POS[1]]
    
    found = run_a_star(start_cell, end_cell, FOG_GRID)
    
    if found:
        full_path = reconstruct_path(end_cell)
        if len(full_path) > 0:
            next_cell = full_path[0]
            next_pos = (next_cell.y, next_cell.x)
            
            # Determine Instruction
            dr = next_cell.y - start_pos[0]
            dc = next_cell.x - start_pos[1]
            instr = None
            if dr == -1: instr = 'U'
            elif dr == 1: instr = 'D'
            elif dc == -1: instr = 'L'
            elif dc == 1: instr = 'R'
            
            return instr, next_pos, FOG_GRID, full_path
            
    return None, start_pos, FOG_GRID, []

# --- VISUALIZATION ---
def visualize_maze_dstar():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Robot Navigation with Fog of War (N={FOG_OF_WAR_N})")
    clock = pygame.time.Clock()
    
    current_robot_pos = START_POS
    robot_path_history = [current_robot_pos] 
    
    # Initial Plan
    instruction, next_pos_coords, fog_grid_state, current_path = \
        generate_and_solve_maze_2(current_robot_pos)

    running = True
    FPS = 60
    MOVE_DELAY_FRAMES = 5

    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- MOVEMENT LOGIC ---
        if current_robot_pos != END_POS:
            if frame_count % MOVE_DELAY_FRAMES == 0:
                if instruction:
                    current_robot_pos = next_pos_coords
                    robot_path_history.append(current_robot_pos)
                    print(f"Moved to {current_robot_pos}")
                    
                    instruction, next_pos_coords, fog_grid_state, current_path = \
                        generate_and_solve_maze_2(current_robot_pos)
                else:
                    pass

        # --- DRAWING ---
        screen.fill(WHITE)

        for r in range(ROWS):
            for c in range(COLS):
                fog_grid_state[r][c].draw(screen, is_fog_of_war_draw=True)
        
        if current_path:
            for cell in current_path:
                cell.draw_path(screen, YELLOW)

        for r, c in robot_path_history:
             fog_grid_state[r][c].draw_path(screen, LIGHT_BLUE)

        fog_grid_state[END_POS[0]][END_POS[1]].draw_path(screen, RED)
        fog_grid_state[current_robot_pos[0]][current_robot_pos[1]].draw_path(screen, GREEN)

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()
    sys.exit()

def setup():
    # INITIAL SETUP
    global GLOBAL_MAZE_GRID, FOG_GRID
    if GLOBAL_MAZE_GRID is None:
        GLOBAL_MAZE_GRID = [[Cell(c, r) for c in range(COLS)] for r in range(ROWS)]
        
        generate_maze(GLOBAL_MAZE_GRID[0][0], GLOBAL_MAZE_GRID)
        
        print(f"Total Walls Removed: {len(REMOVED_WALLS)}")

        FOG_GRID = [[Cell(c, r) for c in range(COLS)] for r in range(ROWS)]
        
        for r in range(ROWS):
            for c in range(COLS):
                cell = FOG_GRID[r][c]
                cell.walls = {'N': False, 'E': False, 'S': False, 'W': False} 
                
                # Hard borders
                if r == 0: cell.walls['N'] = True
                if r == ROWS - 1: cell.walls['S'] = True
                if c == 0: cell.walls['W'] = True
                if c == COLS - 1: cell.walls['E'] = True
                
                # Obstacle
                if is_in_obstacle(r, c):
                     GLOBAL_MAZE_GRID[r][c].walls = {'N':True, 'E':True, 'S':True, 'W':True}
                     cell.walls = {'N':True, 'E':True, 'S':True, 'W':True}
                     cell.known = True

if __name__ == "__main__":
    setup()

    removed_walls = REMOVED_WALLS
    visualize_maze_dstar() #comment this out if you don't want visualization