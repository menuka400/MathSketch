import cv2
import numpy as np
import mediapipe as mp
import groq
import time
import math
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import os
from shapely.geometry import Polygon, LineString, Point
import re

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Groq client with provided API key
try:
    api_key = "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"
    client = groq.Client(api_key=api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    exit(1)

model_name = "deepseek-r1-distill-llama-70b"  # Provided model name (verify with xAI API)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Constants
DRAWING_COLOR = (0, 0, 255)  # Red color for drawing
POINT_COLOR = (0, 255, 0)    # Green color for points
LINE_THICKNESS = 2
POINT_RADIUS = 5
FINGER_TRACK_LENGTH = 64     # Corrected from FINGER_TRACK Herod
SHAPE_DETECTION_THRESHOLD = 10  # Minimum points to detect a shape
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CLEAR_BUTTON = (20, 20, 100, 60)  # (x, y, w, h)
SOLVE_BUTTON = (120, 20, 100, 60)  # (x, y, w, h)
ANNOTATION_COLOR = (255, 255, 0)  # Yellow for annotations
GESTURE_HOLD_TIME = 10  # Frames to confirm gesture

# Drawing variables
finger_points = deque(maxlen=FINGER_TRACK_LENGTH)
shapes = []
numbers = []
operators = []
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
solution_printed = False  # Flag to prevent solution spam

def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def analyze_shape(points):
    """Analyze points to determine geometric shape"""
    if len(points) < SHAPE_DETECTION_THRESHOLD:
        return None
    
    # Convert points to numpy array
    points_array = np.array(points)
    
    # Calculate convex hull
    hull = cv2.convexHull(points_array)
    hull_points = [tuple(p[0]) for p in hull]
    
    # Calculate perimeter and area
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    
    # Avoid division by zero
    if perimeter == 0:
        return None
    
    # Calculate shape complexity using isoperimetric quotient
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Approximate polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    num_vertices = len(approx)
    
    # Create shapely polygon for more analysis
    if len(hull_points) >= 3:
        shape_poly = Polygon(hull_points)
        min_x, min_y, max_x, max_y = shape_poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / height if height > 0 else 0
    else:
        aspect_ratio = 0
    
    # Identify shape
    shape_info = {
        "type": None,
        "vertices": num_vertices,
        "points": hull_points,
        "center": (np.mean(points_array[:, 0]), np.mean(points_array[:, 1])),
        "area": area,
        "perimeter": perimeter
    }
    
    # Triangle
    if num_vertices == 3:
        shape_info["type"] = "triangle"
        
        # Check if it's an equilateral triangle
        sides = [distance(hull_points[i], hull_points[(i+1)%3]) for i in range(3)]
        avg_side = sum(sides) / 3
        if all(abs(side - avg_side) / avg_side < 0.2 for side in sides):
            shape_info["type"] = "equilateral triangle"
        
        # Check if it's a right triangle
        for i in range(3):
            a = distance(hull_points[i], hull_points[(i+1)%3])
            b = distance(hull_points[(i+1)%3], hull_points[(i+2)%3])
            c = distance(hull_points[(i+2)%3], hull_points[i])
            sides = sorted([a, b, c])
            if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) / sides[2]**2 < 0.2:
                shape_info["type"] = "right triangle"
                break
    
    # Rectangle or Square
    elif num_vertices == 4:
        # Calculate sides and angles
        sides = [distance(hull_points[i], hull_points[(i+1)%4]) for i in range(4)]
        avg_side = sum(sides) / 4
        side_variation = max(abs(side - avg_side) / avg_side for side in sides)
        
        if side_variation < 0.2 and 0.8 < aspect_ratio < 1.2:
            shape_info["type"] = "square"
        else:
            shape_info["type"] = "rectangle"
    
    # Circle
    elif circularity > 0.8:
        shape_info["type"] = "circle"
        # Calculate radius
        shape_info["radius"] = math.sqrt(area / math.pi)
    
    # Irregular polygon
    elif num_vertices > 4:
        if num_vertices == 5 and circularity > 0.6:
            shape_info["type"] = "pentagon"
        elif num_vertices == 6 and circularity > 0.6:
            shape_info["type"] = "hexagon"
        else:
            shape_info["type"] = f"polygon_{num_vertices}"

    # Recognize numbers
    if shape_info["type"] is None and recognize_digit(points_array):
        digit = recognize_digit(points_array)
        shape_info["type"] = "number"
        shape_info["value"] = digit
    
    # Recognize operators
    if shape_info["type"] is None:
        operator = recognize_operator(points_array)
        if operator:
            shape_info["type"] = "operator"
            shape_info["value"] = operator
    
    return shape_info

def recognize_digit(points):
    """Basic digit recognition based on shape analysis"""
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    
    if perimeter == 0:
        return None
    
    # Calculate bounding box
    x, y, w, h = cv2.boundingRect(points)
    aspect_ratio = w / h if h > 0 else 0
    
    # Calculate the centroid
    M = cv2.moments(points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Very simple digit recognition based on shape characteristics
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
    """Recognize mathematical operators"""
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
    """Detect if user is showing index finger only or index+middle fingers"""
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
    """Parse detected shapes into a mathematical problem"""
    global shapes
    
    # Conversion factor (cm per pixel) - adjust based on calibration
    PIXEL_TO_CM = 0.2  # Example: 0.2 cm/pixel (calibrate with a known size)
    
    sorted_shapes = sorted(shapes, key=lambda s: s["center"][0])
    
    problem = ""
    for shape in sorted_shapes:
        if shape["type"] == "number":
            problem += shape["value"]
        elif shape["type"] == "operator":
            problem += " " + shape["value"] + " "
        elif "triangle" in shape["type"]:
            points = shape["points"]
            sides = [distance(points[i], points[(i+1)%3]) * PIXEL_TO_CM for i in range(3)]  # Convert to cm
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
            sides = [distance(points[i], points[(i+1)%4]) * PIXEL_TO_CM for i in range(4)]  # Convert to cm
            side_length = sum(sides) / 4
            problem += f"square with side length {round(side_length, 1)} cm"
        elif shape["type"] == "rectangle":
            points = shape["points"]
            sides = [distance(points[i], points[(i+1)%4]) * PIXEL_TO_CM for i in range(4)]  # Convert to cm
            width = min(max(sides[0], sides[2]), max(sides[1], sides[3]))
            height = max(min(sides[0], sides[2]), min(sides[1], sides[3]))
            problem += f"rectangle with width {round(width, 1)} cm and height {round(height, 1)} cm"
        elif shape["type"] == "circle":
            radius = shape["radius"] * PIXEL_TO_CM  # Convert to cm
            problem += f"circle with radius {round(radius, 1)} cm"
    
    return problem

def solve_geometric_problem(problem):
    """Use Groq API to solve the geometric problem"""
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
            max_tokens=1000
        )
        # Post-process to remove any residual LaTeX or tags
        response_text = response.choices[0].message.content
        # Remove LaTeX-like symbols and tags
        response_text = re.sub(r'\\boxed{.*?}', '', response_text)
        response_text = re.sub(r'\\[a-zA-Z]+{.*?}', '', response_text)
        response_text = re.sub(r'\\', '', response_text)
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return response_text.strip()
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return f"Error solving problem: {str(e)}"

def main():
    global finger_points, is_drawing, is_holding, last_point, current_drawing, canvas
    global shapes, math_problem, solution, processing, gesture_counter, last_gesture
    global solution_printed

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Create a status area at the bottom of the screen
    status_area = np.zeros((100, FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Maximize the window
    cv2.namedWindow('Geometric Math Problem Solver', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Geometric Math Problem Solver', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
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
        
        # Initialize gesture for this frame
        current_gesture = "none"
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect current gesture
                current_gesture = detect_gesture(hand_landmarks)
                
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * frame.shape[1])
                y = int(index_finger_tip.y * frame.shape[0])
                
                # Draw circle at finger tip
                cv2.circle(frame, (x, y), POINT_RADIUS, POINT_COLOR, -1)
                
                # Check if finger is touching "Clear" button
                if point_in_rect((x, y), CLEAR_BUTTON):
                    # Clear everything
                    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    current_drawing = []
                    shapes = []
                    math_problem = ""
                    solution = ""
                    solution_printed = False  # Reset flag
                    is_drawing = False
                    is_holding = False
                    last_point = None
                
                # Check if finger is touching "Solve" button and not processing
                elif point_in_rect((x, y), SOLVE_BUTTON) and not processing:
                    processing = True
                    # Process current drawing if any
                    if current_drawing:
                        shape_info = analyze_shape(np.array(current_drawing))
                        if shape_info:
                            shapes.append(shape_info)
                        current_drawing = []
                    
                    # Parse all shapes into a math problem
                    math_problem = parse_math_problem()
                    if math_problem:
                        print(f"Identified problem: {math_problem}")
                        # Solve the problem using Groq API
                        solution = solve_geometric_problem(math_problem)
                        solution_printed = False  # Reset flag for new solution
                    else:
                        solution = "Could not identify a valid geometric problem."
                        solution_printed = False  # Reset flag for error
                    processing = False
                
                # Handle gesture-based drawing
                else:
                    # Add point to drawing history
                    finger_points.append((x, y))
                    
                    # Handle different gestures
                    if current_gesture == "index" and not is_holding:
                        # Single index finger - drawing mode
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
                        # Index + Middle finger - hold drawing
                        if is_drawing and not is_holding:
                            is_holding = True
                            is_drawing = False
                            if len(current_drawing) > SHAPE_DETECTION_THRESHOLD:
                                shape_info = analyze_shape(np.array(current_drawing))
                                if shape_info:
                                    shapes.append(shape_info)
                                    cv2.putText(canvas, shape_info["type"], 
                                              (int(shape_info["center"][0]), int(shape_info["center"][1])), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
                            current_drawing = []
                    
                    elif current_gesture == "other":
                        # Other gesture - reset drawing state
                        if is_drawing:
                            is_drawing = False
                            if len(current_drawing) > SHAPE_DETECTION_THRESHOLD:
                                shape_info = analyze_shape(np.array(current_drawing))
                                if shape_info:
                                    shapes.append(shape_info)
                                    cv2.putText(canvas, shape_info["type"], 
                                              (int(shape_info["center"][0]), int(shape_info["center"][1])), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
                            current_drawing = []
                        is_holding = False
        else:
            # No hands detected - reset drawing state
            if is_drawing:
                is_drawing = False
                if len(current_drawing) > SHAPE_DETECTION_THRESHOLD:
                    shape_info = analyze_shape(np.array(current_drawing))
                    if shape_info:
                        shapes.append(shape_info)
                        cv2.putText(canvas, shape_info["type"], 
                                  (int(shape_info["center"][0]), int(shape_info["center"][1])), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, ANNOTATION_COLOR, 1)
                current_drawing = []
            is_holding = False
        
        # Track gesture changes for stable detection
        if current_gesture != last_gesture:
            gesture_counter = 0
            last_gesture = current_gesture
        else:
            gesture_counter += 1
        
        # Combine frame with canvas
        result = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
        
        # Update status area
        status_area.fill(0)
        
        # Display current gesture
        gesture_text = "No Hand Detected"
        if current_gesture == "index":
            gesture_text = "Drawing Mode (Index Finger)"
        elif current_gesture == "index_middle":
            gesture_text = "Hold Mode (Index + Middle Fingers)"
        elif current_gesture == "other":
            gesture_text = "No Action (Other Gesture)"
        
        cv2.putText(status_area, gesture_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display instructions
        cv2.putText(status_area, "Draw with index finger | Hold drawing with index+middle | Clear/Solve with buttons", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display math problem and solution if available
        if math_problem:
            cv2.putText(status_area, f"Problem: {math_problem[:50]}", 
                       (FRAME_WIDTH // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            if solution:
                y_pos = FRAME_HEIGHT + 20
                solution_lines = solution.split('\n')
                if len(solution_lines) > 0:
                    cv2.putText(status_area, f"Solution: {solution_lines[0][:50]}", 
                               (FRAME_WIDTH // 3, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                    
                    # Print full solution to console only if not printed yet
                    if not solution_printed:
                        print("\n----- SOLUTION -----")
                        print(solution)
                        print("--------------------\n")
                        solution_printed = True
        
        # Combine result with status area
        combined_frame = np.vstack((result, status_area))
        
        # Display the resulting frame
        cv2.imshow('Geometric Math Problem Solver', combined_frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()