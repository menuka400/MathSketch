from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from shapely.geometry import Polygon
import cv2
import numpy as np
import mediapipe as mp
import base64
import threading
import time
import math
from collections import deque
import groq
import re
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# Constants
DRAWING_COLOR = (0, 0, 255)  # Red color for drawing
POINT_COLOR = (0, 255, 0)    # Green color for points
LINE_THICKNESS = 2
POINT_RADIUS = 5
FINGER_TRACK_LENGTH = 64
SHAPE_DETECTION_THRESHOLD = 10  # Minimum points to detect a shape
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CLEAR_BUTTON = (20, 20, 100, 60)  # (x, y, w, h)
SOLVE_BUTTON = (120, 20, 100, 60)  # (x, y, w, h)
ANNOTATION_COLOR = (255, 255, 0)  # Yellow for annotations
DIMENSION_COLOR = (0, 255, 255)  # Cyan for dimension text
GESTURE_HOLD_TIME = 10  # Frames to confirm gesture
PIXEL_TO_CM = 0.2  # Pixel to centimeter conversion factor

# Initialize Groq client
try:
    api_key = os.getenv("GROQ_API_KEY", "")
    client = groq.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    socketio.emit('error', {'message': f'Failed to initialize Groq client: {str(e)}'})

model_name = "deepseek-r1-distill-llama-70b"

# Drawing variables
finger_points = deque(maxlen=FINGER_TRACK_LENGTH)
shapes = []
shapes_lock = threading.Lock()  # Thread-safe access to shapes
is_drawing = False
is_holding = False
last_point = None
current_drawing = []
canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
math_problem = ""
solution = ""
processing = False
gesture_counter = 0
last_gesture = None
current_gesture = "none"
solution_printed = False

# Video capture
camera = None
is_running = False

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python-native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def analyze_shape(points):
    """Analyze points to determine geometric shape and dimensions."""
    if len(points) < SHAPE_DETECTION_THRESHOLD:
        if len(points) >= 2:  # Treat as a line if too few points for a shape
            start = points[0]
            end = points[-1]
            length_cm = distance(start, end) * PIXEL_TO_CM
            center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            return {
                "type": "line",
                "points": [start, end],
                "center": center,
                "length_cm": round(length_cm, 1),
                "dimensions_text": f"{round(length_cm, 1)} cm"
            }
        return None

    points_array = np.array(points)
    hull = cv2.convexHull(points_array)
    hull_points = [tuple(p[0].tolist()) for p in hull]
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)

    if perimeter == 0:
        return None

    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    num_vertices = len(approx)

    if len(hull_points) >= 3:
        shape_poly = Polygon(hull_points)
        min_x, min_y, max_x, max_y = shape_poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height if height > 0 else 0
    else:
        aspect_ratio = 0

    shape_info = {
        "type": None,
        "vertices": num_vertices,
        "points": hull_points,
        "center": (float(np.mean(points_array[:, 0])), float(np.mean(points_array[:, 1]))),
        "area": float(area),
        "perimeter": float(perimeter),
        "dimensions_text": ""
    }

    if num_vertices == 3:
        shape_info["type"] = "triangle"
        sides = [distance(hull_points[i], hull_points[(i+1)%3]) * PIXEL_TO_CM for i in range(3)]
        avg_side = sum(sides) / 3
        shape_info["dimensions_text"] = f"Sides: {round(sides[0], 1)}, {round(sides[1], 1)}, {round(sides[2], 1)} cm"
        if all(abs(side - avg_side) / avg_side < 0.2 for side in sides):
            shape_info["type"] = "equilateral triangle"
            shape_info["dimensions_text"] = f"Side: {round(avg_side, 1)} cm"
        for i in range(3):
            a = distance(hull_points[i], hull_points[(i+1)%3])
            b = distance(hull_points[(i+1)%3], hull_points[(i+2)%3])
            c = distance(hull_points[(i+2)%3], hull_points[i])
            sides_px = sorted([a, b, c])
            if abs(sides_px[0]**2 + sides_px[1]**2 - sides_px[2]**2) / sides_px[2]**2 < 0.2:
                shape_info["type"] = "right triangle"
                sides_cm = [s * PIXEL_TO_CM for s in sides_px]
                shape_info["dimensions_text"] = f"Sides: {round(sides_cm[0], 1)}, {round(sides_cm[1], 1)}, {round(sides_cm[2], 1)} cm"
                break

    elif num_vertices == 4:
        sides = [distance(hull_points[i], hull_points[(i+1)%4]) * PIXEL_TO_CM for i in range(4)]
        avg_side = sum(sides) / 4
        side_variation = max(abs(side - avg_side) / avg_side for side in sides)
        if side_variation < 0.2 and 0.8 < aspect_ratio < 1.2:
            shape_info["type"] = "square"
            shape_info["dimensions_text"] = f"Side: {round(avg_side, 1)} cm"
        else:
            shape_info["type"] = "rectangle"
            width_cm = min(max(sides[0], sides[2]), max(sides[1], sides[3]))
            height_cm = max(min(sides[0], sides[2]), min(sides[1], sides[3]))
            shape_info["dimensions_text"] = f"W: {round(width_cm, 1)} cm, H: {round(height_cm, 1)} cm"

    elif circularity > 0.8:
        shape_info["type"] = "circle"
        radius = math.sqrt(area / math.pi) * PIXEL_TO_CM
        shape_info["radius"] = radius
        shape_info["dimensions_text"] = f"Radius: {round(radius, 1)} cm"

    elif num_vertices > 4:
        if num_vertices == 5 and circularity > 0.6:
            shape_info["type"] = "pentagon"
        elif num_vertices == 6 and circularity > 0.6:
            shape_info["type"] = "hexagon"
        else:
            shape_info["type"] = f"polygon_{num_vertices}"
        sides = [distance(hull_points[i], hull_points[(i+1)%num_vertices]) * PIXEL_TO_CM for i in range(num_vertices)]
        shape_info["dimensions_text"] = f"Sides: {', '.join([str(round(s, 1)) for s in sides])} cm"

    if shape_info["type"] is None and recognize_digit(points_array):
        digit = recognize_digit(points_array)
        shape_info["type"] = "number"
        shape_info["value"] = digit
        shape_info["dimensions_text"] = f"Digit: {digit}"

    if shape_info["type"] is None:
        operator = recognize_operator(points_array)
        if operator:
            shape_info["type"] = "operator"
            shape_info["value"] = operator
            shape_info["dimensions_text"] = f"Operator: {operator}"

    return shape_info

def recognize_digit(points):
    """Basic digit recognition based on shape analysis."""
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)

    if perimeter == 0:
        return None

    x, y, w, h = cv2.boundingRect(points)
    aspect_ratio = w / h if h > 0 else 0

    M = cv2.moments(points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    if 0.7 < circularity < 0.9 and aspect_ratio < 1.2:
        return "0"
    if aspect_ratio < 0.5 and h > w * 2:
        return "1"
    if w > h * 0.5 and w < h * 1.2:
        top_points = sum(1 for p in points if p[0][1] < cy)
        bottom_points = len(points) - top_points
        if top_points > bottom_points * 0.6:
            return "2"
    if w > h * 0.5 and w < h * 1.2 and 0.4 < circularity < 0.7:
        return "3"
    if aspect_ratio > 0.6 and aspect_ratio < 1.2:
        left_points = sum(1 for p in points if p[0][0] < cx)
        if left_points > len(points) * 0.6:
            return "4"
    if w > h * 0.5 and w < h * 1.2 and 0.4 < circularity < 0.7:
        top_points = sum(1 for p in points if p[0][1] < cy)
        if top_points < len(points) * 0.5:
            return "5"
    if 0.5 < circularity < 0.8 and aspect_ratio < 1.2:
        bottom_points = sum(1 for p in points if p[0][1] > cy)
        if bottom_points > len(points) * 0.6:
            return "6"
    if aspect_ratio > 0.5 and aspect_ratio < 1.2:
        top_points = sum(1 for p in points if p[0][1] < cy)
        if top_points > len(points) * 0.3 and top_points < len(points) * 0.5:
            return "7"
    if 0.6 < circularity < 0.9 and aspect_ratio < 1.2:
        return "8"
    if 0.5 < circularity < 0.8 and aspect_ratio < 1.2:
        top_points = sum(1 for p in points if p[0][1] < cy)
        if top_points > len(points) * 0.6:
            return "9"

    return None

def recognize_operator(points):
    """Recognize mathematical operators."""
    x, y, w, h = cv2.boundingRect(points)
    aspect_ratio = w / h if h > 0 else 0

    M = cv2.moments(points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    if 0.7 < aspect_ratio < 1.3:
        center_points = sum(1 for p in points if abs(p[0][0] - cx) < w/4 or abs(p[0][1] - cy) < h/4)
        if center_points > len(points) * 0.7:
            return "+"

    if aspect_ratio > 2.0 and h < 20:
        horizontal_points = sum(1 for p in points if abs(p[0][1] - cy) < h/2)
        if horizontal_points > len(points) * 0.8:
            return "-"

    if 0.7 < aspect_ratio < 1.3:
        diag1_points = sum(1 for p in points if abs((p[0][0] - x) / w - (p[0][1] - y) / h) < 0.3)
        diag2_points = sum(1 for p in points if abs((p[0][0] - x) / w + (p[0][1] - y) / h - 1) < 0.3)
        if (diag1_points + diag2_points) > len(points) * 0.6:
            return "×"

    if aspect_ratio < 0.7:
        diag_points = sum(1 for p in points if abs((p[0][0] - x) / w - (p[0][1] - y) / h) < 0.3)
        if diag_points > len(points) * 0.7:
            return "÷"

    if aspect_ratio > 1.5:
        top_half = [p for p in points if p[0][1] < cy]
        bottom_half = [p for p in points if p[0][1] >= cy]
        if len(top_half) > 10 and len(bottom_half) > 10:
            return "="

    return None

def detect_gesture(hand_landmarks):
    """Detect hand gestures."""
    if not hand_landmarks:
        return "none"

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_folded = ring_tip.y > ring_pip.y
    pinky_folded = pinky_tip.y > pinky_pip.y

    if index_extended and not middle_extended and ring_folded and pinky_folded:
        return "index"
    elif index_extended and middle_extended and ring_folded and pinky_folded:
        return "index_middle"
    else:
        return "other"

def parse_math_problem():
    """Parse detected shapes into a mathematical problem."""
    global shapes

    with shapes_lock:
        sorted_shapes = sorted(shapes, key=lambda s: s["center"][0])

    problem = ""
    for shape in sorted_shapes:
        if shape["type"] == "number":
            problem += shape["value"]
        elif shape["type"] == "operator":
            problem += " " + shape["value"] + " "
        elif "triangle" in shape["type"]:
            points = shape["points"]
            sides = [distance(points[i], points[(i+1)%3]) * PIXEL_TO_CM for i in range(3)]
            if shape["type"] == "equilateral triangle":
                problem += f"equilateral triangle with side length {round(sum(sides)/3, 1)} cm"
            elif shape["type"] == "right triangle":
                max_side = max(sides)
                other_sides = [s for s in sides if s != max_side]
                problem += f"right triangle with sides {round(other_sides[0], 1)}, {round(other_sides[1], 1)} and hypotenuse {round(max_side, 1)} cm"
            else:
                problem += f"triangle with sides {round(sides[0], 1)}, {round(sides[1], 1)}, {round(sides[2], 1)} cm"
        elif shape["type"] == "square":
            points = shape["points"]
            sides = [distance(points[i], points[(i+1)%4]) * PIXEL_TO_CM for i in range(4)]
            side_length = sum(sides) / 4
            problem += f"square with side length {round(side_length, 1)} cm"
        elif shape["type"] == "rectangle":
            points = shape["points"]
            sides = [distance(points[i], points[(i+1)%4]) * PIXEL_TO_CM for i in range(4)]
            width = min(max(sides[0], sides[2]), max(sides[1], sides[3]))
            height = max(min(sides[0], sides[2]), min(sides[1], sides[3]))
            problem += f"rectangle with width {round(width, 1)} cm and height {round(height, 1)} cm"
        elif shape["type"] == "circle":
            radius = shape["radius"] * PIXEL_TO_CM
            problem += f"circle with radius {round(radius, 1)} cm"
        elif shape["type"] == "line":
            problem += f"line with length {shape['length_cm']} cm"

    return problem

def solve_geometric_problem(problem):
    """Use Groq API to solve the geometric problem."""
    if not problem:
        return "No valid geometric problem detected."

    prompt = f"""
    Solve the following geometric math problem step-by-step:
    {problem}

    Provide calculations for:
    - Area
    - Perimeter/Circumference
    - Longest side (if applicable)
    - Diagonal (if applicable)

    Format the response as plain text for terminal display. Use simple notation (e.g., '55.8 cm^2' for area, '33.8 cm' for perimeter). Avoid LaTeX, markdown, or tags like <think>. Show each step clearly with headings (e.g., 'Area', 'Perimeter'). Include units (cm or cm^2) and round to one decimal place. End with a summary of all results.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a geometric mathematics expert. Provide accurate, clear solutions in plain text for terminal display."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            timeout=30
        )
        response_text = response.choices[0].message.content
        print("Groq API response:", response_text)
        response_text = re.sub(r'\\boxed{.*?}', '', response_text)
        response_text = re.sub(r'\\[a-zA-Z]+{.*?}', '', response_text)
        response_text = re.sub(r'\\', '', response_text)
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return response_text.strip()
    except Exception as e:
        print(f"Error with Groq API: {e}")
        socketio.emit('error', {'message': f'Groq API error: {str(e)}'})
        return f"Error solving problem: {str(e)}"

def gen_frames():
    """Generate video frames with hand tracking, drawing, and dimension display."""
    global camera, is_running, current_gesture, shapes, is_drawing, is_holding
    global last_point, current_drawing, canvas, math_problem, solution, processing

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        socketio.emit('error', {'message': 'Could not open camera'})
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while is_running:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to capture frame")
                socketio.emit('error', {'message': 'Failed to capture frame'})
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
            results = hands.process(frame_rgb)

            # Draw buttons
            cv2.rectangle(frame, (CLEAR_BUTTON[0], CLEAR_BUTTON[1]),
                         (CLEAR_BUTTON[0] + CLEAR_BUTTON[2], CLEAR_BUTTON[1] + CLEAR_BUTTON[3]),
                         (0, 255, 255), -1)
            cv2.putText(frame, "Clear", (CLEAR_BUTTON[0] + 10, CLEAR_BUTTON[1] + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.rectangle(frame, (SOLVE_BUTTON[0], SOLVE_BUTTON[1]),
                         (SOLVE_BUTTON[0] + SOLVE_BUTTON[2], SOLVE_BUTTON[1] + SOLVE_BUTTON[3]),
                         (0, 255, 0), -1)
            cv2.putText(frame, "Solve", (SOLVE_BUTTON[0] + 10, SOLVE_BUTTON[1] + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            current_gesture = "none"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_gesture = detect_gesture(hand_landmarks)

                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * frame.shape[1])
                    y = int(index_finger_tip.y * frame.shape[0])

                    cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1)

                    if point_in_rect((x, y), CLEAR_BUTTON):
                        with shapes_lock:
                            canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                            current_drawing = []
                            shapes = []
                            math_problem = ""
                            solution = ""
                            is_drawing = False
                            is_holding = False
                            last_point = None

                    elif point_in_rect((x, y), SOLVE_BUTTON) and not processing:
                        processing = True
                        if current_drawing:
                            shape_info = analyze_shape(np.array(current_drawing))
                            if shape_info:
                                with shapes_lock:
                                    shapes.append(shape_info)
                                    cv2.putText(canvas, shape_info["dimensions_text"],
                                               (int(shape_info["center"][0]), int(shape_info["center"][1]) + 20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIMENSION_COLOR, 1)
                            current_drawing = []

                        math_problem = parse_math_problem()
                        if math_problem:
                            solution = solve_geometric_problem(math_problem)
                        else:
                            solution = "Could not identify a valid geometric problem."
                        processing = False

                    else:
                        finger_points.append((x, y))

                        if current_gesture == "index" and not is_holding:
                            if not is_drawing:
                                is_drawing = True
                                last_point = (x, y)
                                current_drawing.append([x, y])
                            else:
                                dist = distance((x, y), last_point)
                                if dist > 5:
                                    cv2.line(canvas, last_point, (x, y), DRAWING_COLOR, LINE_THICKNESS)
                                    last_point = (x, y)
                                    current_drawing.append([x, y])

                        elif current_gesture == "index_middle":
                            if is_drawing and not is_holding:
                                is_holding = True
                                is_drawing = False
                                if len(current_drawing) >= 2:
                                    shape_info = analyze_shape(np.array(current_drawing))
                                    if shape_info:
                                        with shapes_lock:
                                            shapes.append(shape_info)
                                            cv2.putText(canvas, shape_info["type"],
                                                       (int(shape_info["center"][0]), int(shape_info["center"][1])),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
                                            cv2.putText(canvas, shape_info["dimensions_text"],
                                                       (int(shape_info["center"][0]), int(shape_info["center"][1]) + 20),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIMENSION_COLOR, 1)
                                current_drawing = []

                        elif current_gesture == "other":
                            if is_drawing:
                                is_drawing = False
                                if len(current_drawing) >= 2:
                                    shape_info = analyze_shape(np.array(current_drawing))
                                    if shape_info:
                                        with shapes_lock:
                                            shapes.append(shape_info)
                                            cv2.putText(canvas, shape_info["type"],
                                                       (int(shape_info["center"][0]), int(shape_info["center"][1])),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
                                            cv2.putText(canvas, shape_info["dimensions_text"],
                                                       (int(shape_info["center"][0]), int(shape_info["center"][1]) + 20),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIMENSION_COLOR, 1)
                                current_drawing = []
                            is_holding = False

            # Draw dimensions for all existing shapes
            with shapes_lock:
                for shape in shapes:
                    if shape["dimensions_text"]:
                        cv2.putText(canvas, shape["dimensions_text"],
                                   (int(shape["center"][0]), int(shape["center"][1]) + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIMENSION_COLOR, 1)

            result = cv2.addWeighted(frame, 1, canvas, 0.5, 0)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', result)
            if not ret:
                print("Error: Failed to encode frame")
                socketio.emit('error', {'message': 'Failed to encode frame'})
                continue
            frame_bytes = buffer.tobytes()

            # Convert shapes to JSON-serializable format
            with shapes_lock:
                serializable_shapes = convert_numpy_types(shapes)

            # Send frame and data to all clients
            try:
                socketio.emit('gesture_update', {
                    'image': base64.b64encode(frame_bytes).decode('utf-8'),
                    'gesture': current_gesture,
                    'problem': math_problem,
                    'solution': solution,
                    'shapes': serializable_shapes
                })
                socketio.sleep(0.1)
            except Exception as e:
                print(f"Error emitting gesture_update: {e}")
                socketio.emit('error', {'message': f'Failed to emit frame: {str(e)}'})

            time.sleep(0.033)  # ~30 FPS

    finally:
        if camera:
            camera.release()
            camera = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_camera')
def handle_start_camera():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=gen_frames).start()
    else:
        socketio.emit('error', {'message': 'Camera is already running'})

@socketio.on('stop_camera')
def handle_stop_camera():
    global is_running, camera
    is_running = False
    if camera:
        camera.release()
        camera = None

@socketio.on('clear_canvas')
def handle_clear_canvas():
    global canvas, current_drawing, shapes, math_problem, solution
    with shapes_lock:
        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        current_drawing = []
        shapes = []
        math_problem = ""
        solution = ""

@socketio.on('solve_problem')
def handle_solve_problem():
    global math_problem, solution, shapes, processing
    if processing:
        socketio.emit('error', {'message': 'Already processing a problem'})
        return
    processing = True
    try:
        math_problem = parse_math_problem()
        if math_problem:
            solution = solve_geometric_problem(math_problem)
        else:
            solution = "Could not identify a valid geometric problem."
        print("Solution to emit:", solution)
        socketio.emit('gesture_update', {
            'gesture': current_gesture,
            'problem': math_problem,
            'solution': solution,
            'shapes': convert_numpy_types(shapes)
        })
        socketio.sleep(0.1)
    except Exception as e:
        print(f"Error solving problem: {e}")
        socketio.emit('error', {'message': f'Error solving problem: {str(e)}'})
    finally:
        processing = False

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
