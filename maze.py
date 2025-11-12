import pygame
import random
from collections import deque
import heapq

# --- 1. CONFIGURATION ---
WIDTH = 250
HEIGHT = 250
CELL_SIZE = 40  
COLS = WIDTH // CELL_SIZE
ROWS = HEIGHT // CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0) # Start
RED = (255, 0, 0)   # End
YELLOW = (255, 255, 0) # Final Path
LIGHT_BLUE = (173, 216, 230) # Explored Nodes

# Start and End points (row, col coordinates)
# Start: Bottom-Left
START = (ROWS - 1, 0) # (5, 0) for 6x6 grid
# End: Top-Right
END = (0, COLS - 1)   # (0, 5) for 6x6 grid

# --- 2. CELL CLASS ---
class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y # x=col, y=row
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False
        self.parent = None 
        self.distance = float('inf') 

    # Needed for the tie-breaker fix: Python needs to be able to compare Cell objects
    # if the first two elements (distance and counter) in the heapq tuple are equal.
    # We define a method to let the Cell be compared based on its coordinates.
    def __lt__(self, other):
        # Comparison logic: compare by row (y), then by col (x)
        return (self.y, self.x) < (other.y, other.x)

    def draw(self, screen):
        x, y = self.x * CELL_SIZE, self.y * CELL_SIZE
        
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
        # Draw a small inner rectangle for the path/visited cell
        rect = pygame.Rect(x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, color, rect)

# --- 3. MAZE GENERATION UTILITIES ---
def check_cell(x, y, grid):
    """Returns the cell at (x, y) if it's in bounds, otherwise returns None."""
    if 0 <= x < COLS and 0 <= y < ROWS:
        return grid[y][x]
    return None

def remove_wall(current, next_cell):
    """Removes the wall between two adjacent cells."""
    dx = current.x - next_cell.x
    dy = current.y - next_cell.y
    
    if dx == 1:  # next_cell is West of current
        current.walls['W'] = False
        next_cell.walls['E'] = False
    elif dx == -1: # next_cell is East of current
        current.walls['E'] = False
        next_cell.walls['W'] = False
    
    if dy == 1:  # next_cell is North of current
        current.walls['N'] = False
        next_cell.walls['S'] = False
    elif dy == -1: # next_cell is South of current
        current.walls['S'] = False
        next_cell.walls['N'] = False

def generate_maze(start_cell, grid):
    """Generates the maze using Recursive Backtracking (DFS)."""
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


# --- 4. DIJKSTRA'S ALGORITHM IMPLEMENTATION ---
def get_neighbors(cell, grid):
    """Returns a list of accessible (open-path) neighbors for a cell."""
    neighbors = []
    r, c = cell.y, cell.x # row, col

    # Directions: (dr, dc, wall_key)
    moves = [
        (-1, 0, 'N'), # North (dr=-1, row decreases -> Up)
        (1, 0, 'S'),  # South (dr=1, row increases -> Down)
        (0, 1, 'E'),  # East (dc=1, col increases -> Right)
        (0, -1, 'W')  # West (dc=-1, col decreases -> Left)
    ]

    for dr, dc, wall_key in moves:
        nr, nc = r + dr, c + dc
        next_cell = check_cell(nc, nr, grid)
        
        # Check if the neighbor exists AND the wall between them is removed
        if next_cell and not cell.walls[wall_key]:
            neighbors.append(next_cell)
            
    return neighbors

def dijkstra_search(grid, start_pos, end_pos):
    """
    Implements Dijkstra's algorithm to find the shortest path, collecting 
    the expansion order.
    """
    start_cell = grid[start_pos[0]][start_pos[1]]
    end_cell = grid[end_pos[0]][end_pos[1]]
    
    # Counter for tie-breaking: prevents TypeError when distances are equal.
    tie_breaker = 0 
    
    # Priority Queue: stores (distance, counter, cell)
    pq = [(0, tie_breaker, start_cell)]
    start_cell.distance = 0
    
    visited_nodes = {}
    expansion_order = []
    
    while pq:
        # Extract distance, counter (ignored), and cell
        current_distance, _, current_cell = heapq.heappop(pq)
        
        # Optimization check
        if current_distance > current_cell.distance:
            continue
        
        # Record the expansion order 
        expansion_order.append(current_cell)
        
        # Goal check
        if current_cell == end_cell:
            break
        
        visited_nodes[current_cell] = True

        # Explore neighbors
        for neighbor in get_neighbors(current_cell, grid):
            new_distance = current_distance + 1 # Edge weight is 1
            
            if new_distance < neighbor.distance:
                neighbor.distance = new_distance
                neighbor.parent = current_cell
                
                # Increment the tie-breaker counter
                tie_breaker += 1
                
                # Push the new item with the counter
                heapq.heappush(pq, (new_distance, tie_breaker, neighbor))
                
    return visited_nodes, expansion_order


def reconstruct_path(end_cell):
    """Reconstructs the path from the end_cell back to the start using parent pointers."""
    path = deque()
    current = end_cell
    # Trace back until we hit the start cell (which has no parent)
    while current and current.parent: 
        path.appendleft(current)
        current = current.parent
    if current:
        path.appendleft(current) # Add the start cell
    return path

# --- 5. PATH INSTRUCTION GENERATION (NEW) ---
def generate_move_instructions(path):
    """
    Converts a sequence of Cell objects (the path) into a list of 
    directional instructions (U, D, L, R).
    The path is a deque of Cell objects.
    """
    instructions = []
    
    # Iterate through the path, up to the second-to-last cell
    for i in range(len(path) - 1):
        current_cell = path[i]
        next_cell = path[i+1]
        
        # Calculate the change in coordinates (d_row, d_col)
        # Remember: y = row, x = col
        d_row = next_cell.y - current_cell.y
        d_col = next_cell.x - current_cell.x
        
        move = None
        
        # If moving up (row decreases)
        if d_row == -1 and d_col == 0:
            move = 'U' 
        # If moving down (row increases)
        elif d_row == 1 and d_col == 0:
            move = 'D' 
        # If moving left (col decreases)
        elif d_col == -1 and d_row == 0:
            move = 'L' 
        # If moving right (col increases)
        elif d_col == 1 and d_row == 0:
            move = 'R' 
        
        if move:
            instructions.append(move)
        else:
            # Should not happen if the path is valid
            print(f"Error: Invalid move from ({current_cell.y}, {current_cell.x}) to ({next_cell.y}, {next_cell.x})")

    return instructions

# --- 6. MAIN EXECUTION ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maze Dijkstra Solver with Path Instructions")
    clock = pygame.time.Clock()
    
    # Initialize font for numbering nodes
    font = pygame.font.Font(None, 20) 

    # Create the grid
    grid = [[Cell(c, r) for c in range(COLS)] for r in range(ROWS)]
    
    # Generate the maze
    start_cell_gen = grid[0][0] 
    generate_maze(start_cell_gen, grid)
    
    # Reset tracking variables for the search
    for r in range(ROWS):
        for c in range(COLS):
            grid[r][c].visited = False
            grid[r][c].distance = float('inf')
            grid[r][c].parent = None

    # Run Dijkstra's and unpack the results
    visited_nodes, expansion_order = dijkstra_search(grid, START, END)
    
    # Get the final path (sequence of Cell objects)
    end_cell = grid[END[0]][END[1]]
    path = reconstruct_path(end_cell)

    # Generate and Print Instructions
    move_instructions = generate_move_instructions(path)
    print("--- Shortest Path Instructions (Start to Goal) ---")
    print(move_instructions)
    print(f"Total steps: {len(move_instructions)}")
    print("--------------------------------------------------")
    '''
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # Draw the maze walls
        for r in range(ROWS):
            for c in range(COLS):
                grid[r][c].draw(screen)
        
        # Draw the Dijkstra's explored nodes (light blue)
        for cell in visited_nodes:
            cell.draw_path(screen, LIGHT_BLUE) 
        
        # Draw the final path (yellow)
        for cell in path:
            cell.draw_path(screen, YELLOW)
            
        # Draw the expansion order numbers
        for i, cell in enumerate(expansion_order):
            # We start the count at 1
            label_text = str(i + 1)
            
            # Prepare the surface for the text (Black text)
            text_surface = font.render(label_text, True, BLACK)
            
            # Calculate the position to center the text in the cell
            x = cell.x * CELL_SIZE + (CELL_SIZE - text_surface.get_width()) // 2
            y = cell.y * CELL_SIZE + (CELL_SIZE - text_surface.get_height()) // 2
            
            # Draw the text on the screen
            screen.blit(text_surface, (x, y))
            
        # Highlight Start (GREEN) and End (RED) on top
        grid[START[0]][START[1]].draw_path(screen, GREEN)
        grid[END[0]][END[1]].draw_path(screen, RED)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    '''

if __name__ == "__main__":
    main()