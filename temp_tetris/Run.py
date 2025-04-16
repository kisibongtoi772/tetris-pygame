import threading
from HelloWorld import TetrisGame


tetris_game = TetrisGame()
game_thread = threading.Thread(target=tetris_game.run)
game_thread.start()

