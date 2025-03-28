import pygame
from VoiceCommandRec import VoiceCommandRecognizer
import torch
import os
import threading
import queue
import random

# Tetris piece shapes and their rotations
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # J
    [[1, 1, 1], [0, 0, 1]],  # L
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]   # Z
]

# Colors for shapes
COLORS = [
    (0, 255, 255),  # Cyan
    (255, 255, 0),  # Yellow
    (128, 0, 128),  # Purple
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (0, 255, 0),    # Green
    (255, 0, 0)     # Red
]

class TetrisGame:
    def __init__(self, command_queue):
        self.screen = None
        self.clock = None
        self.model = VoiceCommandRecognizer()
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        voice_model = os.path.join(base_dir, 'voice_model.pth')
        self.model.load_state_dict(torch.load(voice_model))
        self.command_queue = command_queue
        
        # Game board setup
        self.board_width = 10
        self.board_height = 20
        self.block_size = 20
        self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        
        # Game state
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.current_color = (255, 255, 255)
        self.score = 0
        
        # Start with a new piece
        self.new_piece()

    def initialize_pygame(self):
        # Initialize pygame components in main thread
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Voice-Controlled Tetris")
        self.clock = pygame.time.Clock()
        
    def new_piece(self):
        shape_idx = random.randint(0, len(SHAPES) - 1)
        self.current_piece = SHAPES[shape_idx]
        self.current_color = COLORS[shape_idx]
        
        # Start position (centered at top)
        self.current_x = self.board_width // 2 - len(self.current_piece[0]) // 2
        self.current_y = 0
        
        # Game over check
        if not self.valid_position():
            self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
            self.score = 0
    
    def valid_position(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == 0:
                    continue
                    
                board_x = self.current_x + x
                board_y = self.current_y + y
                
                if board_x < 0 or board_x >= self.board_width or board_y >= self.board_height:
                    return False
                    
                if board_y >= 0 and self.board[board_y][board_x] != 0:
                    return False
        return True

    def run(self):
        # Make sure pygame is initialized in the main thread
        self.initialize_pygame()
        
        # Game timing variables
        drop_time = 0
        drop_speed = 500  # time in ms between drops
        last_time = pygame.time.get_ticks()
        
        running = True
        while running:
            # Calculate time since last frame
            now = pygame.time.get_ticks()
            delta_time = now - last_time
            last_time = now
            
            # Add to drop timer
            drop_time += delta_time
            
            # Automatic dropping
            if drop_time > drop_speed:
                self.drop()
                drop_time = 0
                
            # Process voice commands from queue (if any)
            try:
                while not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    self.handle_command(command)
            except queue.Empty:
                pass
            
            # Handle pygame events in main thread
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_left()
                    elif event.key == pygame.K_RIGHT:
                        self.move_right()
                    elif event.key == pygame.K_UP:
                        self.rotate()
                    elif event.key == pygame.K_DOWN:
                        self.drop()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen and draw
            self.screen.fill((0, 0, 0))
            self.draw_board()
            self.draw_piece()
            self.draw_score()
            pygame.display.flip()
            self.clock.tick(60)
    
    def handle_command(self, command):
        # Process voice commands received from another thread
        if command == "move_left":
            self.move_left()
        elif command == "move_right":
            self.move_right()
        elif command == "rotate":
            self.rotate()
        elif command == "drop":
            self.drop()

    def move_left(self):
        self.current_x -= 1
        if not self.valid_position():
            self.current_x += 1

    def move_right(self):
        self.current_x += 1
        if not self.valid_position():
            self.current_x -= 1

    def rotate(self):
        # Transpose and reverse rows to rotate 90°
        rotated = list(zip(*self.current_piece[::-1]))
        rotated = [list(row) for row in rotated]
        
        # Store original piece in case rotation is not valid
        original = self.current_piece
        self.current_piece = rotated
        
        if not self.valid_position():
            self.current_piece = original

    def drop(self):
        self.current_y += 1
        if not self.valid_position():
            self.current_y -= 1
            self.merge_piece()
            self.clear_lines()
            self.new_piece()

    def merge_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == 0:
                    continue
                    
                board_y = self.current_y + y
                board_x = self.current_x + x
                
                if 0 <= board_y < self.board_height and 0 <= board_x < self.board_width:
                    self.board[board_y][board_x] = self.current_color
    
    def clear_lines(self):
        lines_cleared = 0
        for y in range(self.board_height):
            if all(cell != 0 for cell in self.board[y]):
                # Move all lines above down
                for y2 in range(y, 0, -1):
                    self.board[y2] = self.board[y2-1][:]
                # Clear top line
                self.board[0] = [0] * self.board_width
                lines_cleared += 1
        
        # Update score
        if lines_cleared > 0:
            self.score += 100 * (2 ** (lines_cleared - 1))
    
    def draw_board(self):
        board_left = (640 - self.board_width * self.block_size) // 2
        board_top = (480 - self.board_height * self.block_size) // 2
        
        # Draw border
        pygame.draw.rect(
            self.screen, 
            (100, 100, 100),
            [
                board_left - 2, 
                board_top - 2,
                self.board_width * self.block_size + 4, 
                self.board_height * self.block_size + 4
            ], 
            2
        )
        
        # Draw filled blocks
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x] != 0:
                    color = self.board[y][x]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        [
                            board_left + x * self.block_size,
                            board_top + y * self.block_size,
                            self.block_size,
                            self.block_size
                        ]
                    )
                    pygame.draw.rect(
                        self.screen,
                        (200, 200, 200),
                        [
                            board_left + x * self.block_size,
                            board_top + y * self.block_size,
                            self.block_size,
                            self.block_size
                        ],
                        1
                    )
    
    def draw_piece(self):
        if self.current_piece is None:
            return
            
        board_left = (640 - self.board_width * self.block_size) // 2
        board_top = (480 - self.board_height * self.block_size) // 2
        
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell == 0:
                    continue
                    
                pygame.draw.rect(
                    self.screen,
                    self.current_color,
                    [
                        board_left + (self.current_x + x) * self.block_size,
                        board_top + (self.current_y + y) * self.block_size,
                        self.block_size,
                        self.block_size
                    ]
                )
                pygame.draw.rect(
                    self.screen,
                    (200, 200, 200),
                    [
                        board_left + (self.current_x + x) * self.block_size,
                        board_top + (self.current_y + y) * self.block_size,
                        self.block_size,
                        self.block_size
                    ],
                    1
                )
    
    def draw_score(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (20, 20))

# For your main file:
def process_voice_commands(model, command_queue):
    import pyaudio
    import numpy as np
    import librosa
    import time
    
    # Audio recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Sample rate
    CHUNK = 1024  # Frames per buffer
    RECORD_SECONDS = 1.0  # Length of each recording segment
    
    # Command mapping from model predictions to game commands
    commands = ["left", "right", "rotate", "drop"]
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Voice command recognition active. Say commands to control Tetris.")
    
    try:
        while True:
            # Record audio
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            # Convert audio to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize
            
            # Extract spectrogram features
            if len(audio_data) > 0:
                # Generate mel spectrogram (shape will be [n_mels, time])
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data, 
                    sr=RATE,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=96
                )
                
                # Convert to decibels for better feature representation
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Resize to match model's expected input dimensions
                # The model expects input that will result in 32x24x28 features after convolutions
                # This requires an input of size [1, 1, 96, 112]
                padded = np.zeros((96, 112))
                padded[:mel_spec.shape[0], :min(mel_spec.shape[1], 112)] = mel_spec[:, :min(mel_spec.shape[1], 112)]
                mel_spec = padded
                
                # Prepare for model (add batch and channel dimensions)
                model_input = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
                
                # Make prediction
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():  # No need to track gradients
                    output = model(model_input)
                    
                # Get prediction index
                print(output)
                predicted_idx = torch.argmax(output, dim=1).item()
                print(predicted_idx)
                # Energy threshold check to avoid quiet inputs
                energy = np.mean(np.abs(audio_data))
                if energy > 0.01:  # Simple threshold, adjust based on your mic
                    command = commands[predicted_idx]
                    print(f"Detected command: {command}")
                    command_queue.put(command)
            
            time.sleep(0.1)  # Small delay to avoid excessive CPU usage
            
    except KeyboardInterrupt:
        print("Voice command recognition stopped.")
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    command_queue = queue.Queue()
    game = TetrisGame(command_queue)
    
    # Start voice processing in a separate thread
    voice_thread = threading.Thread(target=process_voice_commands, 
                                   args=(game.model, command_queue))
    voice_thread.daemon = True
    voice_thread.start()
    
    # Run the game loop in the main thread
    game.run()