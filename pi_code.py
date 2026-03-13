"""
Dynamic Traffic Light Control System
Two-Stage Operation:
1. Live prediction view (both cameras side-by-side)
2. Press 's' to start traffic control
3. Press 'q' to quit anytime

Usage: python3 traffic_code.py --sim
"""

import cv2
import time
import numpy as np
import sys
from picamera2 import Picamera2
from ultralytics import YOLO
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("GPIO not available")
    GPIO = None

# ==================== MODE SELECTION ====================
USE_LEDS = False

if len(sys.argv) > 1:
    if sys.argv[1] == '--sim' or sys.argv[1] == '--terminal':
        USE_LEDS = False
        print("🚦 Running in TERMINAL mode")
    elif sys.argv[1] == '--led':
        USE_LEDS = True
        print("🚦 Running in LED mode")
    else:
        print("Usage:")
        print("  python3 traffic_code.py --sim    # Terminal visualization")
        print("  python3 traffic_code.py --led    # LED control")
        sys.exit(1)
else:
    print("Usage:")
    print("  python3 traffic_code.py --sim    # Terminal visualization")
    print("  python3 traffic_code.py --led    # LED control")
    sys.exit(1)

# ==================== CONFIGURATION ====================

# Timing Parameters
MIN_GREEN_TIME = 8       # Minimum green time (safety)
MAX_GREEN_TIME = 30      # Maximum green time per phase
MAX_WAIT_TIME = 45       # Maximum wait before forced switch (anti-starvation)
YELLOW_TIME = 3
ALL_RED_TIME = 2

# GPIO Pin Configuration
ROAD_A_PINS = {'red': 17, 'yellow': 27, 'green': 22}
ROAD_B_PINS = {'red': 23, 'yellow': 24, 'green': 25}

# Region of Interest - adjust based on camera view
ROI_ROAD_A = [0.0, 0.0, 1.0, 1.0]  # Full frame initially
ROI_ROAD_B = [0.0, 0.0, 1.0, 1.0]

# YOLO Configuration
NCNN_MODEL_PATH = "yolov8n_ncnn_model"
CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']

# ==================== ANSI COLORS ====================

class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# ==================== GPIO SETUP ====================

def setup_gpio():
    """Initialize GPIO pins for traffic lights"""
    if not USE_LEDS or GPIO is None:
        return
   
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
   
    all_pins = list(ROAD_A_PINS.values()) + list(ROAD_B_PINS.values())
    for pin in all_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
   
    print("✓ GPIO initialized")

def cleanup_gpio():
    """Clean up GPIO"""
    if USE_LEDS and GPIO is not None:
        GPIO.cleanup()

# ==================== TRAFFIC LIGHT CONTROL ====================

def print_traffic_state(road_a_color, road_b_color):
    """Print traffic light state in terminal"""
    print("\n" + "="*50)
    print(f"  ROAD A: {Colors.BOLD}", end="")
    if road_a_color == 'red':
        print(f"{Colors.RED}🔴 RED{Colors.RESET}", end="")
    elif road_a_color == 'yellow':
        print(f"{Colors.YELLOW}🟡 YELLOW{Colors.RESET}", end="")
    elif road_a_color == 'green':
        print(f"{Colors.GREEN}🟢 GREEN{Colors.RESET}", end="")
   
    print(f"     ROAD B: {Colors.BOLD}", end="")
    if road_b_color == 'red':
        print(f"{Colors.RED}🔴 RED{Colors.RESET}")
    elif road_b_color == 'yellow':
        print(f"{Colors.YELLOW}🟡 YELLOW{Colors.RESET}")
    elif road_b_color == 'green':
        print(f"{Colors.GREEN}🟢 GREEN{Colors.RESET}")
    print("="*50)

def set_light(road_pins, color):
    """Set traffic light color"""
    if not USE_LEDS or GPIO is None:
        return
   
    for pin in road_pins.values():
        GPIO.output(pin, GPIO.LOW)
   
    if color in road_pins:
        GPIO.output(road_pins[color], GPIO.HIGH)

def set_all_red():
    """Set both roads to RED"""
    if USE_LEDS:
        set_light(ROAD_A_PINS, 'red')
        set_light(ROAD_B_PINS, 'red')
    else:
        print_traffic_state('red', 'red')

# ==================== CAMERA & YOLO ====================

def initialize_cameras():
    """Initialize both Pi cameras"""
    try:
        print("Initializing cameras...")
        camera_a = Picamera2(0)
        camera_b = Picamera2(1)
       
        config_a = camera_a.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        config_b = camera_b.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
       
        camera_a.configure(config_a)
        camera_b.configure(config_b)
       
        camera_a.start()
        camera_b.start()
       
        # Give cameras more time to warm up (important for avoiding timeout errors)
        print("Waiting for cameras to warm up...")
        time.sleep(4)
        
        # Test capture to verify cameras are working
        test_a = camera_a.capture_array()
        test_b = camera_b.capture_array()
        
        if test_a is not None and test_b is not None:
            print(f"✓ Camera A: {test_a.shape}")
            print(f"✓ Camera B: {test_b.shape}")
            print("✓ Cameras initialized")
        else:
            print("⚠ Warning: Test capture returned None")
        
        return camera_a, camera_b
    except Exception as e:
        print(f"✗ Camera initialization failed: {e}")
        return None, None

def initialize_yolo():
    """Load YOLOv8n NCNN model"""
    try:
        print("Loading YOLO model...")
        model = YOLO(NCNN_MODEL_PATH)
        print(f"✓ YOLOv8n NCNN model loaded")
        return model
    except Exception as e:
        print(f"✗ YOLO loading failed: {e}")
        return None

def apply_roi(frame, roi_percentages):
    """Extract ROI from frame"""
    h, w = frame.shape[:2]
    x = int(roi_percentages[0] * w)
    y = int(roi_percentages[1] * h)
    width = int(roi_percentages[2] * w)
    height = int(roi_percentages[3] * h)
    return frame[y:y+height, x:x+width]

def detect_and_count(model, frame, roi_percentages):
    """
    Run YOLO detection and count vehicles
    Returns: (vehicle_count, annotated_frame)
    """
    if model is None or frame is None:
        return 0, frame
   
    try:
        # Apply ROI
        roi_frame = apply_roi(frame, roi_percentages)
       
        # Run detection
        results = model(roi_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
       
        # Count vehicles
        vehicle_count = 0
        vehicle_types = {}
       
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
               
                if class_name in VEHICLE_CLASSES:
                    vehicle_count += 1
                    vehicle_types[class_name] = vehicle_types.get(class_name, 0) + 1
       
        # Draw bounding boxes
        annotated = results[0].plot()
       
        # Add count overlay
        text = f"Vehicles: {vehicle_count}"
        if vehicle_types:
            types_str = ", ".join([f"{v} {k}" for k, v in vehicle_types.items()])
            text += f" ({types_str})"
       
        cv2.putText(annotated, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        return vehicle_count, annotated
   
    except Exception as e:
        print(f"Detection error: {e}")
        return 0, frame

# ==================== LIVE PREDICTION MODE ====================

def run_live_predictions(camera_a, camera_b, model):
    """
    Show live predictions from both cameras
    Press 's' to start traffic control
    Press 'q' to quit
    """
    print("\n" + "="*60)
    print("LIVE PREDICTION MODE")
    print("="*60)
    print("📹 Showing detections from both cameras")
    print("   Press 'S' to START traffic control")
    print("   Press 'Q' to QUIT")
    print("="*60 + "\n")
   
    cv2.namedWindow('Traffic Detection - Road A (Left) | Road B (Right)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Traffic Detection - Road A (Left) | Road B (Right)', 1280, 480)
   
    fps_start = time.time()
    frame_count = 0
   
    while True:
        try:
            # Capture from both cameras
            frame_a = camera_a.capture_array()
            frame_b = camera_b.capture_array()
            
            # Debug: Check if frames are valid
            if frame_count == 0:
                print(f"DEBUG: Frame A type={type(frame_a)}, shape={frame_a.shape if frame_a is not None else 'None'}")
                print(f"DEBUG: Frame B type={type(frame_b)}, shape={frame_b.shape if frame_b is not None else 'None'}")
            
            # Validate frames
            if frame_a is None or frame_b is None:
                print("Warning: Received None frame from camera")
                time.sleep(0.1)
                continue
                
            if frame_a.size == 0 or frame_b.size == 0:
                print("Warning: Received empty frame from camera")
                time.sleep(0.1)
                continue
           
            # Run detection on both
            count_a, annotated_a = detect_and_count(model, frame_a, ROI_ROAD_A)
            count_b, annotated_b = detect_and_count(model, frame_b, ROI_ROAD_B)
           
            # Add road labels
            cv2.putText(annotated_a, "ROAD A", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(annotated_b, "ROAD B", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
           
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                print(f"FPS: {fps:.1f} | Road A: {count_a} vehicles | Road B: {count_b} vehicles")
           
            # Combine frames side by side
            combined = np.hstack((annotated_a, annotated_b))
           
            # Add instruction text
            cv2.putText(combined, "Press 'S' to start traffic control | Press 'Q' to quit",
                       (10, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
           
            # Display
            cv2.imshow('Traffic Detection - Road A (Left) | Road B (Right)', combined)
           
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n🛑 Quitting...")
                return False
            elif key == ord('s') or key == ord('S'):
                print("\n🚦 Starting traffic control mode...")
                cv2.destroyAllWindows()
                return True
       
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            return False
        except Exception as e:
            print(f"Error in live predictions: {e}")
            time.sleep(0.1)
   
    return False

# ==================== TRAFFIC CONTROL MODE ====================

def draw_traffic_light_overlay(frame, light_color, road_name, countdown=None, vehicle_count=None):
    """
    Draw traffic light indicator overlay on frame
    light_color: 'red', 'yellow', or 'green'
    """
    h, w = frame.shape[:2]
    
    # Color mappings (BGR format for OpenCV)
    color_map = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0)
    }
    bg_color = color_map.get(light_color, (128, 128, 128))
    
    # Draw semi-transparent background for status bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw traffic light circle
    circle_center = (50, 40)
    circle_radius = 25
    cv2.circle(frame, circle_center, circle_radius + 3, (255, 255, 255), 2)  # White border
    cv2.circle(frame, circle_center, circle_radius, bg_color, -1)  # Filled circle
    
    # Draw road name and status
    status_text = f"{road_name}: {light_color.upper()}"
    cv2.putText(frame, status_text, (90, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, bg_color, 2)
    
    # Draw countdown timer if available
    if countdown is not None:
        countdown_text = f"Time: {countdown:.0f}s"
        cv2.putText(frame, countdown_text, (90, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw vehicle count if available
    if vehicle_count is not None:
        count_text = f"Vehicles: {vehicle_count}"
        cv2.putText(frame, count_text, (w - 180, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def capture_and_count_with_frame(camera, model, roi, road_name):
    """Capture, count vehicles, and return both count and annotated frame"""
    try:
        frame = camera.capture_array()
        count, annotated = detect_and_count(model, frame, roi)
       
        # Log count
        vehicle_types = {}
        results = model(apply_roi(frame, roi), conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name in VEHICLE_CLASSES:
                    vehicle_types[class_name] = vehicle_types.get(class_name, 0) + 1
       
        if count > 0:
            types_str = ", ".join([f"{v} {k}" for k, v in vehicle_types.items()])
            print(f"  📷 {road_name}: {count} vehicles ({types_str})")
        else:
            print(f"  📷 {road_name}: {count} vehicles")
       
        return count, annotated
    except Exception as e:
        print(f"  ⚠ Capture failed for {road_name}: {e}")
        return 0, frame if 'frame' in locals() else None

def display_traffic_frame(camera_a, camera_b, model, light_a, light_b, countdown, phase_name, count_a=None, count_b=None):
    """Capture from both cameras and display with traffic light overlays"""
    try:
        # Capture from both cameras
        frame_a = camera_a.capture_array()
        frame_b = camera_b.capture_array()
        
        # Run detection on both
        _, annotated_a = detect_and_count(model, frame_a, ROI_ROAD_A)
        _, annotated_b = detect_and_count(model, frame_b, ROI_ROAD_B)
        
        # Add traffic light overlays
        annotated_a = draw_traffic_light_overlay(annotated_a, light_a, "ROAD A", countdown, count_a)
        annotated_b = draw_traffic_light_overlay(annotated_b, light_b, "ROAD B", countdown, count_b)
        
        # Combine frames side by side
        combined = np.hstack((annotated_a, annotated_b))
        
        # Add phase info at bottom
        cv2.putText(combined, f"Phase: {phase_name} | Press 'Q' to stop",
                   (10, combined.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display
        cv2.imshow('Traffic Control - Road A (Left) | Road B (Right)', combined)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            return False
        return True
    except Exception as e:
        print(f"Display error: {e}")
        return True

def run_phase_with_display(camera_a, camera_b, model, light_a, light_b, duration, phase_name, count_a=None, count_b=None):
    """Run a traffic phase while continuously displaying video with overlays"""
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        remaining = duration - elapsed
        
        if remaining <= 0:
            break
        
        # Display frame with countdown
        if not display_traffic_frame(camera_a, camera_b, model, light_a, light_b, remaining, phase_name, count_a, count_b):
            return False  # User pressed Q
        
        # Small delay to prevent CPU overload
        time.sleep(0.03)
    
    return True

def calculate_proportional_times(count_a, count_b):
    """
    Calculate green times proportional to traffic density.
    Returns (green_time_a, green_time_b)
    """
    total = count_a + count_b
    
    if total == 0:
        # No vehicles - both get minimum time
        return MIN_GREEN_TIME, MIN_GREEN_TIME
    
    # Calculate proportions
    ratio_a = count_a / total
    ratio_b = count_b / total
    
    # Allocate time proportionally within [MIN, MAX] range
    time_range = MAX_GREEN_TIME - MIN_GREEN_TIME
    green_time_a = MIN_GREEN_TIME + time_range * ratio_a
    green_time_b = MIN_GREEN_TIME + time_range * ratio_b
    
    return green_time_a, green_time_b

def display_realtime_frame(camera_a, camera_b, model, light_a, light_b, 
                           count_a, count_b, elapsed, green_time, status_msg):
    """Display frame with real-time traffic info overlay"""
    try:
        # Capture from both cameras
        frame_a = camera_a.capture_array()
        frame_b = camera_b.capture_array()
        
        # Run detection on both
        new_count_a, annotated_a = detect_and_count(model, frame_a, ROI_ROAD_A)
        new_count_b, annotated_b = detect_and_count(model, frame_b, ROI_ROAD_B)
        
        # Add traffic light overlays
        countdown = max(0, green_time - elapsed) if green_time else None
        annotated_a = draw_traffic_light_overlay(annotated_a, light_a, "ROAD A", countdown, new_count_a)
        annotated_b = draw_traffic_light_overlay(annotated_b, light_b, "ROAD B", countdown, new_count_b)
        
        # Combine frames side by side
        combined = np.hstack((annotated_a, annotated_b))
        
        # Add status bar at bottom
        h = combined.shape[0]
        cv2.rectangle(combined, (0, h-50), (combined.shape[1], h), (0, 0, 0), -1)
        cv2.putText(combined, status_msg, (10, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, "Press 'Q' to stop", (combined.shape[1] - 200, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Display
        cv2.imshow('Dynamic Traffic Control', combined)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            return None, None, False
        
        return new_count_a, new_count_b, True
        
    except Exception as e:
        print(f"Display error: {e}")
        return count_a, count_b, True

def run_transition(camera_a, camera_b, model, from_light_a, from_light_b, to_road):
    """Run yellow and all-red transition sequence"""
    
    # Yellow phase
    if to_road == 'B':
        light_a, light_b = 'yellow', 'red'
    else:
        light_a, light_b = 'red', 'yellow'
    
    if USE_LEDS:
        if to_road == 'B':
            set_light(ROAD_A_PINS, 'yellow')
        else:
            set_light(ROAD_B_PINS, 'yellow')
    
    print_traffic_state(light_a, light_b)
    
    start = time.time()
    while time.time() - start < YELLOW_TIME:
        _, _, cont = display_realtime_frame(camera_a, camera_b, model, light_a, light_b, 
                                            0, 0, 0, None, f"YELLOW - Transitioning to Road {to_road}")
        if not cont:
            return False
        time.sleep(0.03)
    
    # All red phase
    set_all_red()
    start = time.time()
    while time.time() - start < ALL_RED_TIME:
        _, _, cont = display_realtime_frame(camera_a, camera_b, model, 'red', 'red',
                                            0, 0, 0, None, "ALL RED - Clearing intersection")
        if not cont:
            return False
        time.sleep(0.03)
    
    return True

def run_traffic_control(camera_a, camera_b, model):
    """
    Real-time dynamic traffic control with continuous monitoring.
    - Standby mode when no vehicles detected
    - Proportional green time based on traffic density
    - Anti-starvation protection
    """
    print("\n" + "="*60)
    print("REAL-TIME DYNAMIC TRAFFIC CONTROL")
    print("="*60)
    print("Press 'Q' in video window or Ctrl+C to stop\n")
    
    # Create window
    cv2.namedWindow('Dynamic Traffic Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Dynamic Traffic Control', 1280, 480)
    
    # State tracking
    current_green = None  # None = standby, 'A' or 'B' = active
    phase_start = None
    current_green_time = MIN_GREEN_TIME
    last_switch_time = time.time()
    cycle_count = 0
    
    try:
        while True:
            # Capture and detect on both cameras
            frame_a = camera_a.capture_array()
            frame_b = camera_b.capture_array()
            
            count_a, _ = detect_and_count(model, frame_a, ROI_ROAD_A)
            count_b, _ = detect_and_count(model, frame_b, ROI_ROAD_B)
            
            total_vehicles = count_a + count_b
            
            # ========== STANDBY MODE ==========
            if current_green is None:
                if total_vehicles == 0:
                    # Stay in standby
                    if USE_LEDS:
                        set_all_red()
                    _, _, cont = display_realtime_frame(
                        camera_a, camera_b, model, 'red', 'red',
                        count_a, count_b, 0, None,
                        "STANDBY - No vehicles detected. Waiting..."
                    )
                    if not cont:
                        break
                    time.sleep(0.03)
                    continue
                else:
                    # Vehicles detected - start active mode
                    cycle_count += 1
                    print(f"\n{'='*60}")
                    print(f"CYCLE {cycle_count} - Vehicles detected, starting control")
                    print(f"  Road A: {count_a} vehicles | Road B: {count_b} vehicles")
                    print(f"{'='*60}")
                    
                    # Give green to road with more vehicles (or A if equal)
                    if count_b > count_a:
                        current_green = 'B'
                    else:
                        current_green = 'A'
                    
                    phase_start = time.time()
                    _, current_green_time = calculate_proportional_times(count_a, count_b) if current_green == 'B' else (None, calculate_proportional_times(count_a, count_b)[0])
                    current_green_time = calculate_proportional_times(count_a, count_b)[0] if current_green == 'A' else calculate_proportional_times(count_a, count_b)[1]
                    
                    print(f"  → Road {current_green} gets GREEN for {current_green_time:.1f}s")
                    
                    if USE_LEDS:
                        if current_green == 'A':
                            set_light(ROAD_A_PINS, 'green')
                            set_light(ROAD_B_PINS, 'red')
                        else:
                            set_light(ROAD_B_PINS, 'green')
                            set_light(ROAD_A_PINS, 'red')
                    
                    continue
            
            # ========== ACTIVE MODE ==========
            elapsed = time.time() - phase_start
            light_a = 'green' if current_green == 'A' else 'red'
            light_b = 'green' if current_green == 'B' else 'red'
            
            # Calculate current proportional times
            green_time_a, green_time_b = calculate_proportional_times(count_a, count_b)
            target_time = green_time_a if current_green == 'A' else green_time_b
            
            # Status message
            remaining = max(0, target_time - elapsed)
            status = f"Road {current_green} GREEN | A:{count_a} B:{count_b} | Time: {elapsed:.1f}s / {target_time:.1f}s"
            
            # Display
            new_a, new_b, cont = display_realtime_frame(
                camera_a, camera_b, model, light_a, light_b,
                count_a, count_b, elapsed, target_time, status
            )
            if not cont:
                break
            
            # Update counts if we got new detections
            if new_a is not None:
                count_a, count_b = new_a, new_b
            
            # ========== DECISION LOGIC ==========
            should_switch = False
            switch_reason = ""
            
            # Check minimum time
            if elapsed < MIN_GREEN_TIME:
                # Must wait minimum time
                pass
            # Check maximum wait time (anti-starvation)
            elif elapsed >= MAX_WAIT_TIME:
                should_switch = True
                switch_reason = "Maximum time reached"
            # Check if current green time is done
            elif elapsed >= target_time:
                should_switch = True
                switch_reason = f"Proportional time ({target_time:.1f}s) completed"
            
            # ========== SWITCH LIGHTS ==========
            if should_switch:
                other_road = 'B' if current_green == 'A' else 'A'
                print(f"\n🔄 SWITCHING: {switch_reason}")
                print(f"  Road A: {count_a} | Road B: {count_b}")
                
                # Check if returning to standby
                if total_vehicles == 0:
                    print("  → No vehicles, returning to STANDBY")
                    if not run_transition(camera_a, camera_b, model, light_a, light_b, other_road):
                        break
                    current_green = None
                    continue
                
                # Calculate new times
                green_time_a, green_time_b = calculate_proportional_times(count_a, count_b)
                new_green_time = green_time_b if other_road == 'B' else green_time_a
                
                print(f"  → Road {other_road} gets GREEN for {new_green_time:.1f}s")
                
                # Run transition
                if not run_transition(camera_a, camera_b, model, light_a, light_b, other_road):
                    break
                
                # Start new green phase
                current_green = other_road
                current_green_time = new_green_time
                phase_start = time.time()
                cycle_count += 1
                
                if USE_LEDS:
                    if current_green == 'A':
                        set_light(ROAD_A_PINS, 'green')
                        set_light(ROAD_B_PINS, 'red')
                    else:
                        set_light(ROAD_B_PINS, 'green')
                        set_light(ROAD_A_PINS, 'red')
                
                print_traffic_state(
                    'green' if current_green == 'A' else 'red',
                    'green' if current_green == 'B' else 'red'
                )
            
            time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Traffic control stopped by user")
    finally:
        print("\n🛑 Shutting down traffic control...")
        set_all_red()
        cv2.destroyAllWindows()

# ==================== MAIN ====================

def main():
    """Main program"""
    print("="*60)
    print("DYNAMIC TRAFFIC LIGHT CONTROL SYSTEM")
    print("="*60)
    print(f"Mode: {'LED Hardware' if USE_LEDS else 'Terminal Visualization'}")
    print("="*60)
   
    # Initialize
    setup_gpio()
    camera_a, camera_b = initialize_cameras()
    model = initialize_yolo()
   
    if camera_a is None or camera_b is None or model is None:
        print("✗ Initialization failed")
        return
   
    try:
        # Stage 1: Live predictions
        start_traffic = run_live_predictions(camera_a, camera_b, model)
       
        # Stage 2: Traffic control (if 's' was pressed)
        if start_traffic:
            set_all_red()
            time.sleep(1)
            run_traffic_control(camera_a, camera_b, model)
   
    finally:
        print("\n🛑 Shutting down...")
        set_all_red()
        if camera_a:
            camera_a.stop()
        if camera_b:
            camera_b.stop()
        cleanup_gpio()
        cv2.destroyAllWindows()
        print("✓ Shutdown complete")

if __name__ == "__main__":
    main()