import cv2 as cv
import mediapipe as mp

import numpy as np
import random
import time
import os

from PIL import Image
from io import BytesIO
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import base64
import tensorflow as tf


from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D



model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(20, 127, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    # LSTM(units=3, input_shape=(1,10)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(250, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load the weights
model.load_weights('./best.weights.h5')

base_options = python.BaseOptions(
    model_asset_path="./hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


asl = {"TV": 0, "after": 1, "airplane": 2, "all": 3, "alligator": 4, "animal": 5, "another": 6, "any": 7, "apple": 8, "arm": 9, "aunt": 10, "awake": 11, "backyard": 12, "bad": 13, "balloon": 14, "bath": 15, "because": 16, "bed": 17, "bedroom": 18, "bee": 19, "before": 20, "beside": 21, "better": 22, "bird": 23, "black": 24, "blow": 25, "blue": 26, "boat": 27, "book": 28, "boy": 29, "brother": 30, "brown": 31, "bug": 32, "bye": 33, "callonphone": 34, "can": 35, "car": 36, "carrot": 37, "cat": 38, "cereal": 39, "chair": 40, "cheek": 41, "child": 42, "chin": 43, "chocolate": 44, "clean": 45, "close": 46, "closet": 47, "cloud": 48, "clown": 49, "cow": 50, "cowboy": 51, "cry": 52, "cut": 53, "cute": 54, "dad": 55, "dance": 56, "dirty": 57, "dog": 58, "doll": 59, "donkey": 60, "down": 61, "drawer": 62, "drink": 63, "drop": 64, "dry": 65, "dryer": 66, "duck": 67, "ear": 68, "elephant": 69, "empty": 70, "every": 71, "eye": 72, "face": 73, "fall": 74, "farm": 75, "fast": 76, "feet": 77, "find": 78, "fine": 79, "finger": 80, "finish": 81, "fireman": 82, "first": 83, "fish": 84, "flag": 85, "flower": 86, "food": 87, "for": 88, "frenchfries": 89, "frog": 90, "garbage": 91, "gift": 92, "giraffe": 93, "girl": 94, "give": 95, "glasswindow": 96, "go": 97, "goose": 98, "grandma": 99, "grandpa": 100, "grass": 101, "green": 102, "gum": 103, "hair": 104, "happy": 105, "hat": 106, "hate": 107, "have": 108, "haveto": 109, "head": 110, "hear": 111, "helicopter": 112, "hello": 113, "hen": 114, "hesheit": 115, "hide": 116, "high": 117, "home": 118, "horse": 119, "hot": 120, "hungry": 121, "icecream": 122, "if": 123, "into": 124, "jacket": 125, "jeans": 126, "jump": 127, "kiss": 128, "kitty": 129, "lamp": 130, "later": 131, "like": 132, "lion": 133, "lips": 134, "listen": 135, "look": 136, "loud": 137, "mad": 138, "make": 139, "man": 140, "many": 141, "milk": 142, "minemy": 143, "mitten": 144, "mom": 145, "moon": 146, "morning": 147, "mouse": 148, "mouth": 149, "nap": 150, "napkin": 151, "night": 152, "no": 153, "noisy": 154, "nose": 155, "not": 156, "now": 157, "nuts": 158, "old": 159, "on": 160, "open": 161, "orange": 162, "outside": 163, "owie": 164, "owl": 165, "pajamas": 166, "pen": 167, "pencil": 168, "penny": 169, "person": 170, "pig": 171, "pizza": 172, "please": 173, "police": 174, "pool": 175, "potty": 176, "pretend": 177, "pretty": 178, "puppy": 179, "puzzle": 180, "quiet": 181, "radio": 182, "rain": 183, "read": 184, "red": 185, "refrigerator": 186, "ride": 187, "room": 188, "sad": 189, "same": 190, "say": 191, "scissors": 192, "see": 193, "shhh": 194, "shirt": 195, "shoe": 196, "shower": 197, "sick": 198, "sleep": 199, "sleepy": 200, "smile": 201, "snack": 202, "snow": 203, "stairs": 204, "stay": 205, "sticky": 206, "store": 207, "story": 208, "stuck": 209, "sun": 210, "table": 211, "talk": 212, "taste": 213, "thankyou": 214, "that": 215, "there": 216, "think": 217, "thirsty": 218, "tiger": 219, "time": 220, "tomorrow": 221, "tongue": 222, "tooth": 223, "toothbrush": 224, "touch": 225, "toy": 226, "tree": 227, "uncle": 228, "underwear": 229, "up": 230, "vacuum": 231, "wait": 232, "wake": 233, "water": 234, "wet": 235, "weus": 236, "where": 237, "white": 238, "who": 239, "why": 240, "will": 241, "wolf": 242, "yellow": 243, "yes": 244, "yesterday": 245, "yourself": 246, "yucky": 247, "zebra": 248, "zipper": 249}
asl_keys = list(asl.keys())
word_list=["TV", "smile", "boat", "thankyou", "head", "up"]
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "https://signly-iota.vercel.app/",      # ← Add your actual URL
            "https://*.vercel.app",             # ← Allows all Vercel preview deployments
            "http://18.224.214.141"             # ← Add AWS EC2 public IP
        ]
    }
})

socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://signly-iota.vercel.app/",              # ← Add your actual URL
    "https://*.vercel.app",
    "http://18.224.214.141"             # ← Add AWS EC2 public IP
])

output=np.zeros((20,126))
indexVector=np.zeros((20,1))
indexVector=np.arange(20)

sentence = random.sample(word_list, 3)
current_word=sentence[0]
current_idx=0
answers = ['incorrect' for _ in range(0, 3)]

@app.route('/')
def index():
    return "WebSocket server running"

@app.route('/hint')
def hint():
    global current_word
    return {"data": current_word}

@socketio.on('connect')
def handle_connect():
    socketio.emit('receive_word', {'message': sentence, 'answers': answers})

@socketio.on('skip_word')
def handle_skip(data):
    global sentence
    global answers
    global current_idx
    global current_word

    # Update sentence and answers from client data (skip comes from client)
    sentence = data.get('message', sentence)
    answers = data.get('answers', answers)

    # If all words are done (correct or skipped), start a new sentence
    if answers.count('correct') == len(answers) or answers.count('skipped') == len(answers):
        print("All words were correctly signed or skipped")
        sentence = random.sample(word_list, 3)
        answers = ['incorrect'] * len(sentence)
        current_idx = 0
        current_word = sentence[0]
    else:
        # Find the next index that still needs to be signed
        for i in range(len(answers)):
            if answers[i] not in ['correct', 'skipped']:
                current_idx = i
                break
        current_word = sentence[current_idx]

    # Emit updated state back to clients
    socketio.emit('receive_word', {'message': sentence, 'answers': answers})

# Add a frame counter to control prediction frequency
frame_counter = 0
PREDICTION_INTERVAL = 2  # Process every 2nd frame

@socketio.on('send_frame')
def handle_frame(data):
    global output, indexVector, sentence, answers, current_word, current_idx, frame_counter

    try:
        frame_counter += 1
        if frame_counter % PREDICTION_INTERVAL != 0:
            return  # Skip this frame

        # Decode frame
        img_data = base64.b64decode(data['frame'].split(',')[1])
        img = np.array(Image.open(BytesIO(img_data)))
        frame = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = hand_landmarker.detect(mp_image)

        landmark_coords = [0.0] * 126

        if result.hand_landmarks:
            for h_idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[h_idx][0].category_name
                offset = 0 if handedness == "Left" else 63

                i = offset
                for lm in landmarks:
                    landmark_coords[i] = lm.x
                    landmark_coords[i + 1] = lm.y
                    landmark_coords[i + 2] = lm.z
                    i += 3

            # Sliding window
            output = output[1:, :]
            output = np.vstack([output, np.array(landmark_coords).reshape(1, -1)])

            # Add index column
            model_input = np.column_stack((indexVector, output))
            model_input = model_input.reshape(1, 20, 127, 1)

            predictions = model.predict(model_input, verbose=0)
            predicted_word = asl_keys[np.argmax(predictions)]

            print(f"PREDICTION: {predicted_word} ({np.max(predictions):.3f})")

            if predicted_word == current_word:
                # Mark current index as correct and pause briefly
                answers[current_idx] = 'correct'

                # If all words are now correct, choose a new sentence
                if answers.count('correct') == len(sentence):
                    sentence = random.sample(word_list, 3)
                    current_idx = 0
                    current_word = sentence[0]
                    answers = ['incorrect'] * len(sentence)
                else:
                    # Advance to the next index that hasn't been completed
                    next_idx = (current_idx + 1) % len(sentence)
                    while answers[next_idx] == 'correct':
                        next_idx = (next_idx + 1) % len(sentence)
                    current_idx = next_idx
                    current_word = sentence[current_idx]

            socketio.emit('receive_word', {
                'message': sentence,
                'answers': answers
            })

    except Exception as e:
        print("FRAME ERROR:", e)
        output = np.zeros((20, 126))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)