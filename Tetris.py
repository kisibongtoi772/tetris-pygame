import random
import pygame
import torch

from kafka import KafkaConsumer

import json
import threading
import queue
import time

voice_command = None  # Global to hold the latest voice result
command_queue = queue.Queue()

def kafka_consumer_thread():
    try:
        consumer = KafkaConsumer(
            'tetris-commands',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='tetris-game',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Kafka consumer connected and listening for commands")
        
        for message in consumer:
            command = message.value.get('command')
            if command:
                command_queue.put(command)
                print(f"Received command: {command}")
    except Exception as e:
        print(f"Kafka consumer error: {e}")
    finally:
        print("Kafka consumer stopped")
# Start Kafka consumer in a background thread
def start_kafka_consumer():
    try:
        kafka_thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
        kafka_thread.start()
        return kafka_thread
    except Exception as e:
        print(f"Failed to start Kafka consumer thread: {e}")
        return None

"""
10 x 20 grid
play_height = 2 * play_width

tetriminos:
    0 - S - green
    1 - Z - red
    2 - I - cyan
    3 - O - yellow
    4 - J - blue
    5 - L - orange
    6 - T - purple
"""
pygame.init()
pygame.font.init()

# global variables

col = 10  # 10 columns
row = 20  # 20 rows
s_width = 800  # window width
s_height = 750  # window height
play_width = 300  # play window width; 300/10 = 30 width per block
play_height = 600  # play window height; 600/20 = 20 height per block
block_size = 30  # size of block

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height - 50

filepath = './highscore.txt'
fontpath = './arcade.ttf'
fontpath_mario = './mario.ttf'

# shapes formats

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['.....',
      '..0..',
      '..0..',
      '..0..',
      '..0..'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


# class to represent each of the pieces


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]  # choose color from the shape_color list
        self.rotation = 0  # chooses the rotation according to index


# initialise the grid
def create_grid(locked_pos={}):
    grid = [[(0, 0, 0) for x in range(col)] for y in range(row)]  # grid represented rgb tuples

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                color = locked_pos[
                    (x, y)]  # get the value color (r,g,b) from the locked_positions dictionary using key (x,y)
                grid[y][x] = color  # set grid position to color

    return grid


def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]  # get the desired rotated shape from piece

    '''
    e.g.
       ['.....',
        '.....',
        '..00.',
        '.00..',
        '.....']
    '''
    for i, line in enumerate(shape_format):  # i gives index; line gives string
        row = list(line)  # makes a list of char from string
        for j, column in enumerate(row):  # j gives index of char; column gives char
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)  # offset according to the input given with dot and zero

    return positions


# checks if current position of piece in grid is valid
def valid_space(piece, grid):
    # makes a 2D list of all the possible (x,y)
    accepted_pos = [[(x, y) for x in range(col) if grid[y][x] == (0, 0, 0)] for y in range(row)]
    # removes sub lists and puts (x,y) in one list; easier to search
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] >= 0:
                return False
    return True


# check if piece is out of board
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


# chooses a shape randomly from shapes list
def get_shape():
    return Piece(5, 0, random.choice(shapes))


# draws text in the middle
def draw_text_middle(text, size, color, surface):
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(fontpath, size)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width()/2), top_left_y + play_height/2 - (label.get_height()/2)))


# draws the lines of the grid for the game
def draw_grid(surface):
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        # draw grey horizontal lines
        pygame.draw.line(surface, grid_color, (top_left_x, top_left_y + i * block_size),
                         (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(col):
            # draw grey vertical lines
            pygame.draw.line(surface, grid_color, (top_left_x + j * block_size, top_left_y),
                             (top_left_x + j * block_size, top_left_y + play_height))


# clear a row when it is filled
def clear_rows(grid, locked):
    # need to check if row is clear then shift every other row above down one
    increment = 0
    for i in range(len(grid) - 1, -1, -1):      # start checking the grid backwards
        grid_row = grid[i]                      # get the last row
        if (0, 0, 0) not in grid_row:           # if there are no empty spaces (i.e. black blocks)
            increment += 1
            # add positions to remove from locked
            index = i                           # row index will be constant
            for j in range(len(grid_row)):
                try:
                    del locked[(j, i)]          # delete every locked element in the bottom row
                except ValueError:
                    continue

    # shift every row one step down
    # delete filled bottom row
    # add another empty row on the top
    # move down one step
    if increment > 0:
        # sort the locked list according to y value in (x,y) and then reverse
        # reversed because otherwise the ones on the top will overwrite the lower ones
        for key in sorted(list(locked), key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:                       # if the y value is above the removed index
                new_key = (x, y + increment)    # shift position to down
                locked[new_key] = locked.pop(key)

    return increment


# draws the upcoming piece
def draw_next_shape(piece, surface):
    font = pygame.font.Font(fontpath, 30)
    label = font.render('Next shape', 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (start_x + j*block_size, start_y + i*block_size, block_size, block_size), 0)

    surface.blit(label, (start_x, start_y - 30))

    # pygame.display.update()


# draws the content of the window
def draw_window(surface, grid, score=0, last_score=0):
    surface.fill((0, 0, 0))  # fill the surface with black

    pygame.font.init()  # initialise font
    font = pygame.font.Font(fontpath_mario, 65)
    label = font.render('TETRIS', 1, (255, 255, 255))  # initialise 'Tetris' text with white

    surface.blit(label, ((top_left_x + play_width / 2) - (label.get_width() / 2), 30))  # put surface on the center of the window

    # current score
    font = pygame.font.Font(fontpath, 30)
    label = font.render('SCORE   ' + str(score) , 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    surface.blit(label, (start_x, start_y + 200))

    # last score
    label_hi = font.render('HIGHSCORE   ' + str(last_score), 1, (255, 255, 255))

    start_x_hi = top_left_x - 240
    start_y_hi = top_left_y + 200

    surface.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

    # draw content of the grid
    for i in range(row):
        for j in range(col):
            # pygame.draw.rect()
            # draw a rectangle shape
            # rect(Surface, color, Rect, width=0) -> Rect
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)

    # draw vertical and horizontal grid lines
    draw_grid(surface)

    # draw rectangular border around play area
    border_color = (255, 255, 255)
    pygame.draw.rect(surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4)

    border_color = (255, 255, 255)
    pygame.draw.rect(surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4)

    # Show confirmation message if awaiting confirmation
    global pending_command, awaiting_confirmation
    if awaiting_confirmation and pending_command:
        # Create a semi-transparent overlay
        overlay = pygame.Surface((play_width, play_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))  # Black with 60% opacity
        surface.blit(overlay, (top_left_x, top_left_y))
        
        # Draw confirmation box
        box_width, box_height = 260, 120
        box_x = top_left_x + play_width//2 - box_width//2
        box_y = top_left_y + play_height//2 - box_height//2
        
        # Background and border
        pygame.draw.rect(surface, (50, 50, 50), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(surface, (255, 255, 255), (box_x, box_y, box_width, box_height), 2)
        
        # Title
        font = pygame.font.Font(fontpath, 20)
        title = f"Confirm {pending_command}"
        title_text = font.render(title, True, (255, 255, 255))
        surface.blit(title_text, (box_x + box_width//2 - title_text.get_width()//2, box_y + 20))
        
        # Instructions
        inst = "Say Yes or No"
        inst_text = font.render(inst, True, (255, 200, 0))
        surface.blit(inst_text, (box_x + box_width//2 - inst_text.get_width()//2, box_y + 60))
        
        # Timer
        time_left = max(0, confirmation_timeout - (time.time() - pending_command_time))
        timer = f"{int(time_left)}s"  # Show just whole seconds with 's' suffix
        timer_text = font.render(timer, True, (200, 200, 200))
        surface.blit(timer_text, (box_x + box_width//2 - timer_text.get_width()//2, box_y + 90))


# update the score txt file with high score
def update_score(new_score):
    score = get_max_score()

    with open(filepath, 'w') as file:
        if new_score > score:
            file.write(str(new_score))
        else:
            file.write(str(score))


# get the high score from the file
def get_max_score():
    with open(filepath, 'r') as file:
        lines = file.readlines()        # reads all the lines and puts in a list
        score = int(lines[0].strip())   # remove \n

    return score

pending_command = None
pending_command_time = 0
commands_requiring_confirmation = ["pause", "speed"]
confirmed_commands = ["yes", "no"]
confirmation_timeout = 7 
is_paused = False
awaiting_confirmation = False  # Add this new flag to track confirmation state

def main(window):
    locked_positions = {}
    create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.35
    level_time = 0
    score = 0
    last_score = get_max_score()
    global pending_command, pending_command_time, is_paused, awaiting_confirmation

    while run:
        # Check if pending command has timed out
        if pending_command and time.time() - pending_command_time > confirmation_timeout:
            print(f"Command {pending_command} timed out waiting for confirmation")
            pending_command = None
            awaiting_confirmation = False  # Reset confirmation state
        
        # need to constantly make new grid as locked positions always change
        grid = create_grid(locked_positions)

        # Only update game state if not paused AND not awaiting confirmation
        if not is_paused and not awaiting_confirmation:
            # helps run the same on every computer
            # add time since last tick() to fall_time
            fall_time += clock.get_rawtime()  # returns in milliseconds
            level_time += clock.get_rawtime()

            clock.tick()  # updates clock

            if level_time/1000 > 5:    # make the difficulty harder every 10 seconds
                level_time = 0
                if fall_speed > 0.15:   # until fall speed is 0.15
                    fall_speed -= 0.005

            if fall_time / 1000 > fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not valid_space(current_piece, grid) and current_piece.y > 0:
                    current_piece.y -= 1
                    # since only checking for down - either reached bottom or hit another piece
                    # need to lock the piece position
                    # need to generate new piece
                    change_piece = True
        else:
            # Still tick the clock when paused, but don't update game state
            clock.tick()

        try:
            while not command_queue.empty():
                command = command_queue.get_nowait()
                
                # Handle confirmation flow
                if pending_command:
                    if command in confirmed_commands:
                        if command == "yes":
                            print(f"Confirmed command: {pending_command}")
                            # Execute the confirmed command
                            if pending_command == "pause":
                                is_paused = not is_paused
                                print(f"Game {'paused' if is_paused else 'resumed'}")
                            elif pending_command == "speed":
                                # Increase game speed by reducing fall_speed
                                fall_speed = max(0.1, fall_speed - 0.1)
                                print(f"Speed increased - fall_speed now: {fall_speed}")
                        elif command == "no":
                            print(f"Rejected command: {pending_command}")
                        
                        # Clear pending command and awaiting flag
                        pending_command = None
                        awaiting_confirmation = False
                    # When there's a pending command, ignore all non-confirmation commands
                    continue
                
                # Process new commands
                if command in commands_requiring_confirmation:
                    pending_command = command
                    pending_command_time = time.time()
                    awaiting_confirmation = True  # Set awaiting confirmation state
                    print(f"Command '{command}' requires confirmation. Say 'yes' to confirm or 'no' to cancel.")
                elif command in confirmed_commands:
                    # Ignore confirmation commands when nothing is pending
                    print(f"No command waiting for confirmation")
                elif not is_paused and not awaiting_confirmation:  # Only process movement commands if game active
                    # Process regular commands only when not paused and not awaiting confirmation
                    if command == "left":
                        current_piece.x -= 1
                        if not valid_space(current_piece, grid):
                            current_piece.x += 1
                            
                    elif command == "right":
                        current_piece.x += 1
                        if not valid_space(current_piece, grid):
                            current_piece.x -= 1
                            
                    elif command == "down":
                        current_piece.y += 1
                        if not valid_space(current_piece, grid):
                            current_piece.y -= 1
                            
                    elif command == "rotation_right":
                        current_piece.rotation = (current_piece.rotation + 1) % len(current_piece.shape)
                        if not valid_space(current_piece, grid):
                            current_piece.rotation = (current_piece.rotation - 1) % len(current_piece.shape)
                            
                    elif command == "rotation_left":
                        current_piece.rotation = (current_piece.rotation - 1) % len(current_piece.shape)
                        if not valid_space(current_piece, grid):
                            current_piece.rotation = (current_piece.rotation + 1) % len(current_piece.shape)
                            
                    elif command == "quit":
                        run = False
                        
        except Exception as e:
            print(f"Error processing command: {e}")
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1  # move x position left
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1  # move x position right
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1

                elif event.key == pygame.K_DOWN:
                    # move shape down
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

                elif event.key == pygame.K_UP:
                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

        piece_pos = convert_shape_format(current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = current_piece.color

        if change_piece:  # if the piece is locked
            for pos in piece_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color       # add the key and value in the dictionary
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += clear_rows(grid, locked_positions) * 10    # increment score by 10 for every row cleared
            update_score(score)

            if last_score < score:
                last_score = score

        draw_window(window, grid, score, last_score)
        draw_next_shape(next_piece, window)
        pygame.display.update()

        if check_lost(locked_positions):
            run = False

    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    pygame.display.update()
    pygame.time.delay(2000)  # wait for 2 seconds
    if check_lost(locked_positions):
        run = False

    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    pygame.display.update()
    pygame.time.delay(2000)


def main_menu(window):
    run = True

    while run:
        draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                main(window)  # After this returns, continue showing the menu
    
    # Only quit pygame when exiting the application
    pygame.quit()

command_queue = queue.Queue()

# Kafka consumer thread function
def kafka_consumer_thread():
    try:
        consumer = KafkaConsumer(
            'tetris-commands',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='tetris-game',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Kafka consumer connected and listening for commands")
        
        for message in consumer:
            command = message.value.get('command')
            if command:
                command_queue.put(command)
                print(f"Received command: {command}")
    except Exception as e:
        print(f"Kafka consumer error: {e}")
    finally:
        print("Kafka consumer stopped")

if __name__ == '__main__':
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Tetris')
    start_kafka_consumer()
    main_menu(win)  # start game
