# ✋ Gesture-Based Geometry Problem Solver

![Geometry Problem Solver](https://img.shields.io/badge/Version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📹 Demo Video

Check out the [demonstration video](https://drive.google.com/file/d/10gpoovLWfpcszAF3m-YF4D4QVoyQS21a/view?usp=sharing) to see the application in action!
## 🚀 Overview

The **Gesture-Based Geometry Problem Solver** is an interactive web application that allows users to draw geometric shapes using hand gestures and solves mathematical problems related to these shapes in real-time. Using computer vision and AI, the application recognizes shapes, mathematical operators, and numerals drawn in the air, making math learning intuitive and engaging.

## ✨ Features

- **Real-time hand tracking** using MediaPipe
- **Shape recognition** for triangles, squares, rectangles, circles, and more
- **Automatic problem solving** with detailed step-by-step solutions
- **Interactive UI** with real-time feedback
- **Gesture controls:**
  - Draw with index finger
  - Hold/confirm shapes with index + middle fingers
  - Clear canvas and solve problems with on-screen buttons

## 🛠️ Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript, Socket.IO
- **Backend**: Flask, Flask-SocketIO
- **Computer Vision**: MediaPipe, OpenCV
- **Shape Analysis**: Shapely, NumPy
- **AI Solution Generation**: Groq AI API (deepseek-r1-distill-llama-70b)

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam
- Internet connection (for Groq API access)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gesture-geometry-solver.git
   cd gesture-geometry-solver
   ```

2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key**
   - Get your API key from [Groq](https://console.groq.com)
   - Replace the placeholder API key in the code or set as environment variable:
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   - Navigate to `http://localhost:5000` in your web browser

## 📝 Usage Guide

1. **Start the camera** by clicking the "Start Camera" button
2. **Draw shapes** with your index finger
3. **Confirm shapes** by extending both your index and middle finger
4. **Clear the canvas** with the "Clear Canvas" button when needed
5. **Solve geometry problems** by clicking the "Solve Problem" button
6. **View the solution** in the right panel

## 📊 Supported Shapes and Problems

- **Basic Shapes**: Triangles, squares, rectangles, circles
- **Special Triangles**:right triangles
- **Calculations**: Area, perimeter/circumference, side lengths, diagonals

## 🔍 How It Works

1. **Hand Tracking**: MediaPipe identifies hand landmarks in real-time
2. **Gesture Recognition**: The system identifies drawing vs. holding gestures
3. **Shape Analysis**: OpenCV and Shapely analyze the drawn points to identify shapes
4. **Problem Formulation**: Detected shapes are analyzed to formulate geometric problems
5. **Solution Generation**: Groq API processes the problem and generates step-by-step solutions
6. **Real-time Feedback**: Socket.IO provides instant updates to the user interface

# ✍️ Geometric Calculation Engine 📐

The heart of this project is its **calculation engine**, which transforms hand-drawn shapes and mathematical expressions into solvable geometric problems. Using computer vision and AI, it accurately detects, parses, and solves drawings in real-time, making math interactive and fun! 🚀

## 🎯 How It Works

The calculation process turns your hand-drawn shapes (e.g., triangles, squares,) into precise geometric or arithmetic solutions. Here's the magic in four steps:

- **✋ Shape Detection**: 
  - Your hand gestures are tracked via webcam to capture drawn points.
  - The system identifies shapes (e.g., triangles, circles) using advanced computer vision techniques.
  - Example: A wobbly triangle is recognized as an "equilateral triangle" with side length 5.2 cm.

- **📝 Problem Parsing**: 
  - Detected shapes are converted into a mathematical problem.
  - Measurements are scaled (pixels to centimeters) for real-world accuracy.
  - Example: A drawn triangle becomes "equilateral triangle with side length 5.2 cm,"

- **🧠 AI-Powered Solving**: 
  - The parsed problem is sent to a powerful AI model (via Groq API) to compute results.
  - For shapes, it calculates **area**, **perimeter**, **longest side**, or **diagonal**.
  - Example: For a triangle, the AI returns:
    
      Area: 11.7 cm²
      Perimeter: 15.6 cm
      Longest side: 5.2 cm

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
---

<p align="center">
  Made with ❤️ by M.H.Jayasuriya
</p>
