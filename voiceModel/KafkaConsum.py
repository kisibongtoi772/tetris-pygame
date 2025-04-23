from kafka import KafkaConsumer
import json
from Thread import tetris_game

consumer = KafkaConsumer('voice-commands',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='earliest',
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

for message in consumer:
    command = message.value['command']
    if command == 'move_left':
        tetris_game.move_left()
    elif command == 'move_right':
        tetris_game.move_right()
    elif command == 'rotate':
        tetris_game.rotate()
    elif command == 'drop':
        tetris_game.drop()