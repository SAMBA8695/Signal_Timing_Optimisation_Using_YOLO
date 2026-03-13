"""
Dynamic Traffic Light Control System — WITH KAFKA
Pipeline: Camera → YOLO Detection → Kafka → Traffic Controller

Kafka Streams:
  - vehicle-detection-events : per-frame detection data from each camera
  - congestion-metrics        : computed congestion levels & green-time decisions

Usage:
  python3 traffic_change_v2.py --sim   # Terminal / simulation mode
  python3 traffic_change_v2.py --led   # LED hardware mode

Kafka must be running locally (localhost:9092) before starting this script.
To spin up quickly: docker-compose up -d  (see README or use a local broker)
"""

import cv2
import time
import json
import uuid
import threading
import numpy as np
import sys
from datetime import datetime

# ── Picamera2 / GPIO (Raspberry Pi only) ────────────────────────────────────
from picamera2 import Picamera2
from ultralytics import YOLO

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("GPIO not available — running without hardware LEDs")
    GPIO = None

# ── Kafka ────────────────────────────────────────────────────────────────────
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    print("⚠  kafka-python not installed. Run:  pip install kafka-python")
    KAFKA_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

USE_LEDS = False

if len(sys.argv) > 1:
    if sys.argv[1] in ("--sim", "--terminal"):
        USE_LEDS = False
        print("🚦 Running in TERMINAL mode")
    elif sys.argv[1] == "--led":
        USE_LEDS = True
        print("🚦 Running in LED mode")
    else:
        print("Usage:")
        print("  python3 traffic_change_v2.py --sim")
        print("  python3 traffic_change_v2.py --led")
        sys.exit(1)
else:
    print("Usage:")
    print("  python3 traffic_change_v2.py --sim")
    print("  python3 traffic_change_v2.py --led")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Timing
MIN_GREEN_TIME  = 8
MAX_GREEN_TIME  = 30
MAX_WAIT_TIME   = 45
YELLOW_TIME     = 3
ALL_RED_TIME    = 2

# GPIO
ROAD_A_PINS = {"red": 17, "yellow": 27, "green": 22}
ROAD_B_PINS = {"red": 23, "yellow": 24, "green": 25}

# ROI
ROI_ROAD_A = [0.0, 0.0, 1.0, 1.0]
ROI_ROAD_B = [0.0, 0.0, 1.0, 1.0]

# YOLO
NCNN_MODEL_PATH     = "yolov8n_ncnn_model"
CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES     = ["car", "motorcycle", "bus", "truck"]

# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9092"

# Topic for raw per-frame detections from each camera
TOPIC_DETECTION = "vehicle-detection-events"

# Topic for aggregated congestion metrics & controller decisions
TOPIC_CONGESTION = "congestion-metrics"


# ══════════════════════════════════════════════════════════════════════════════
#  KAFKA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_producer() -> "KafkaProducer | None":
    """Create a JSON-serialising Kafka producer; returns None if unavailable."""
    if not KAFKA_AVAILABLE:
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",             # wait for leader + replicas to ack
            retries=3,
            linger_ms=10,           # small batching window (ms)
        )
        print(f"✓ Kafka producer connected → {KAFKA_BOOTSTRAP}")
        return producer
    except NoBrokersAvailable:
        print(f"⚠  Kafka broker not reachable at {KAFKA_BOOTSTRAP}. "
              "Running without Kafka (events won't be streamed).")
        return None


def publish_detection_event(producer, road: str, count: int,
                             vehicle_types: dict, frame_id: str) -> None:
    """
    Publish a single vehicle-detection event to Kafka.

    Schema  (vehicle-detection-events):
    {
      "event_id"     : str   – UUID for this event
      "timestamp"    : str   – ISO-8601 UTC
      "road"         : str   – "A" or "B"
      "frame_id"     : str   – shared frame UUID per capture cycle
      "vehicle_count": int
      "vehicle_types": {class: count, ...}
    }
    """
    if producer is None:
        return

    payload = {
        "event_id":      str(uuid.uuid4()),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "road":          road,
        "frame_id":      frame_id,
        "vehicle_count": count,
        "vehicle_types": vehicle_types,
    }
    # Key by road so events from the same road land on the same partition
    producer.send(TOPIC_DETECTION, key=f"road-{road}", value=payload)


def publish_congestion_metric(producer, count_a: int, count_b: int,
                               green_road: str, green_time: float,
                               cycle: int) -> None:
    """
    Publish an aggregated congestion metric to Kafka.

    Schema  (congestion-metrics):
    {
      "event_id"         : str
      "timestamp"        : str
      "cycle"            : int
      "count_a"          : int
      "count_b"          : int
      "total_vehicles"   : int
      "congestion_level" : str   – "none" | "low" | "medium" | "high"
      "green_road"       : str   – "A" | "B" | "none"
      "green_time_s"     : float
      "ratio_a"          : float – fraction of traffic on road A
      "ratio_b"          : float
    }
    """
    if producer is None:
        return

    total = count_a + count_b
    ratio_a = count_a / total if total else 0.0
    ratio_b = count_b / total if total else 0.0

    if total == 0:
        level = "none"
    elif total < 5:
        level = "low"
    elif total < 15:
        level = "medium"
    else:
        level = "high"

    payload = {
        "event_id":        str(uuid.uuid4()),
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "cycle":           cycle,
        "count_a":         count_a,
        "count_b":         count_b,
        "total_vehicles":  total,
        "congestion_level": level,
        "green_road":      green_road or "none",
        "green_time_s":    round(green_time, 2),
        "ratio_a":         round(ratio_a, 3),
        "ratio_b":         round(ratio_b, 3),
    }
    producer.send(TOPIC_CONGESTION, key=f"cycle-{cycle}", value=payload)


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI COLOURS
# ══════════════════════════════════════════════════════════════════════════════

class Colors:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"


# ══════════════════════════════════════════════════════════════════════════════
#  GPIO
# ══════════════════════════════════════════════════════════════════════════════

def setup_gpio():
    if not USE_LEDS or GPIO is None:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in list(ROAD_A_PINS.values()) + list(ROAD_B_PINS.values()):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    print("✓ GPIO initialised")

def cleanup_gpio():
    if USE_LEDS and GPIO is not None:
        GPIO.cleanup()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAFFIC LIGHT CONTROL (terminal + LED)
# ══════════════════════════════════════════════════════════════════════════════

def print_traffic_state(road_a_color, road_b_color):
    print("\n" + "="*50)
    print(f"  ROAD A: {Colors.BOLD}", end="")
    _print_color(road_a_color)
    print(f"     ROAD B: {Colors.BOLD}", end="")
    _print_color(road_b_color, newline=True)
    print("="*50)

def _print_color(color, newline=False):
    mapping = {
        "red":    f"{Colors.RED}🔴 RED{Colors.RESET}",
        "yellow": f"{Colors.YELLOW}🟡 YELLOW{Colors.RESET}",
        "green":  f"{Colors.GREEN}🟢 GREEN{Colors.RESET}",
    }
    end = "\n" if newline else ""
    print(mapping.get(color, color), end=end)

def set_light(road_pins, color):
    if not USE_LEDS or GPIO is None:
        return
    for pin in road_pins.values():
        GPIO.output(pin, GPIO.LOW)
    if color in road_pins:
        GPIO.output(road_pins[color], GPIO.HIGH)

def set_all_red():
    if USE_LEDS:
        set_light(ROAD_A_PINS, "red")
        set_light(ROAD_B_PINS, "red")
    else:
        print_traffic_state("red", "red")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA & YOLO
# ══════════════════════════════════════════════════════════════════════════════

def initialize_cameras():
    try:
        print("Initialising cameras…")
        cam_a = Picamera2(0)
        cam_b = Picamera2(1)
        cfg   = {"size": (640, 480), "format": "RGB888"}
        cam_a.configure(cam_a.create_preview_configuration(main=cfg))
        cam_b.configure(cam_b.create_preview_configuration(main=cfg))
        cam_a.start(); cam_b.start()
        print("Waiting for cameras to warm up…")
        time.sleep(4)
        test_a = cam_a.capture_array()
        test_b = cam_b.capture_array()
        if test_a is not None and test_b is not None:
            print(f"✓ Camera A: {test_a.shape}")
            print(f"✓ Camera B: {test_b.shape}")
        return cam_a, cam_b
    except Exception as e:
        print(f"✗ Camera init failed: {e}")
        return None, None

def initialize_yolo():
    try:
        print("Loading YOLO model…")
        model = YOLO(NCNN_MODEL_PATH)
        print("✓ YOLOv8n NCNN model loaded")
        return model
    except Exception as e:
        print(f"✗ YOLO loading failed: {e}")
        return None

def apply_roi(frame, roi):
    h, w = frame.shape[:2]
    x, y = int(roi[0]*w), int(roi[1]*h)
    return frame[y:y+int(roi[3]*h), x:x+int(roi[2]*w)]

def detect_and_count(model, frame, roi):
    """Run YOLO; return (count, vehicle_types_dict, annotated_frame)."""
    if model is None or frame is None:
        return 0, {}, frame
    try:
        roi_frame = apply_roi(frame, roi)
        results   = model(roi_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        vehicle_count = 0
        vehicle_types = {}
        for result in results:
            for box in result.boxes:
                name = model.names[int(box.cls[0])]
                if name in VEHICLE_CLASSES:
                    vehicle_count += 1
                    vehicle_types[name] = vehicle_types.get(name, 0) + 1

        annotated = results[0].plot()
        text = f"Vehicles: {vehicle_count}"
        if vehicle_types:
            text += " (" + ", ".join(f"{v} {k}" for k, v in vehicle_types.items()) + ")"
        cv2.putText(annotated, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vehicle_count, vehicle_types, annotated
    except Exception as e:
        print(f"Detection error: {e}")
        return 0, {}, frame


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_traffic_light_overlay(frame, light_color, road_name,
                                countdown=None, vehicle_count=None):
    h, w  = frame.shape[:2]
    color_map = {"red": (0,0,255), "yellow": (0,255,255), "green": (0,255,0)}
    bg    = color_map.get(light_color, (128,128,128))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,80), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.circle(frame, (50,40), 28, (255,255,255), 2)
    cv2.circle(frame, (50,40), 25, bg, -1)

    cv2.putText(frame, f"{road_name}: {light_color.upper()}", (90,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bg, 2)
    if countdown is not None:
        cv2.putText(frame, f"Time: {countdown:.0f}s", (90,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if vehicle_count is not None:
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (w-180,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return frame


def display_realtime_frame(camera_a, camera_b, model,
                            light_a, light_b,
                            count_a, count_b,
                            elapsed, green_time, status_msg,
                            producer=None, cycle=0):
    """
    Capture → detect → publish to Kafka → display.
    Returns (new_count_a, new_count_b, continue_flag).
    """
    try:
        frame_a = camera_a.capture_array()
        frame_b = camera_b.capture_array()

        new_a, types_a, ann_a = detect_and_count(model, frame_a, ROI_ROAD_A)
        new_b, types_b, ann_b = detect_and_count(model, frame_b, ROI_ROAD_B)

        # ── Kafka: vehicle detection events ─────────────────────────────────
        frame_id = str(uuid.uuid4())   # shared ID ties road-A & road-B events
        publish_detection_event(producer, "A", new_a, types_a, frame_id)
        publish_detection_event(producer, "B", new_b, types_b, frame_id)

        # Overlays
        countdown = max(0, green_time - elapsed) if green_time else None
        ann_a = draw_traffic_light_overlay(ann_a, light_a, "ROAD A", countdown, new_a)
        ann_b = draw_traffic_light_overlay(ann_b, light_b, "ROAD B", countdown, new_b)

        combined = np.hstack((ann_a, ann_b))

        h = combined.shape[0]
        cv2.rectangle(combined, (0, h-50), (combined.shape[1], h), (0,0,0), -1)
        cv2.putText(combined, status_msg, (10, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(combined, "Press 'Q' to stop",
                    (combined.shape[1]-200, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)

        cv2.imshow("Dynamic Traffic Control", combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            return None, None, False

        return new_a, new_b, True

    except Exception as e:
        print(f"Display error: {e}")
        return count_a, count_b, True


# ══════════════════════════════════════════════════════════════════════════════
#  TIMING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_proportional_times(count_a, count_b):
    total = count_a + count_b
    if total == 0:
        return MIN_GREEN_TIME, MIN_GREEN_TIME
    rng = MAX_GREEN_TIME - MIN_GREEN_TIME
    return (MIN_GREEN_TIME + rng * count_a / total,
            MIN_GREEN_TIME + rng * count_b / total)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE PREDICTION MODE  (unchanged logic, now passes producer)
# ══════════════════════════════════════════════════════════════════════════════

def run_live_predictions(camera_a, camera_b, model, producer=None):
    print("\n" + "="*60)
    print("LIVE PREDICTION MODE")
    print("="*60)
    print("📹 Showing detections from both cameras")
    print("   Press 'S' to START traffic control")
    print("   Press 'Q' to QUIT")
    print("="*60 + "\n")

    cv2.namedWindow("Traffic Detection — Road A | Road B", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Detection — Road A | Road B", 1280, 480)

    fps_start   = time.time()
    frame_count = 0

    while True:
        try:
            frame_a = camera_a.capture_array()
            frame_b = camera_b.capture_array()

            if frame_a is None or frame_b is None:
                time.sleep(0.1); continue

            count_a, types_a, ann_a = detect_and_count(model, frame_a, ROI_ROAD_A)
            count_b, types_b, ann_b = detect_and_count(model, frame_b, ROI_ROAD_B)

            # ── Kafka: publish preview detections ───────────────────────────
            fid = str(uuid.uuid4())
            publish_detection_event(producer, "A", count_a, types_a, fid)
            publish_detection_event(producer, "B", count_b, types_b, fid)

            cv2.putText(ann_a, "ROAD A", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(ann_b, "ROAD B", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                print(f"FPS: {fps:.1f} | Road A: {count_a} | Road B: {count_b}")

            combined = np.hstack((ann_a, ann_b))
            cv2.putText(combined, "Press 'S' to start | Press 'Q' to quit",
                        (10, combined.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.imshow("Traffic Detection — Road A | Road B", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\n🛑 Quitting…"); return False
            elif key in (ord("s"), ord("S")):
                print("\n🚦 Starting traffic control…")
                cv2.destroyAllWindows(); return True

        except KeyboardInterrupt:
            return False
        except Exception as e:
            print(f"Live preview error: {e}")
            time.sleep(0.1)

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_transition(camera_a, camera_b, model, to_road, producer=None, cycle=0):
    """Yellow → All-red transition sequence."""
    light_a = "yellow" if to_road == "B" else "red"
    light_b = "yellow" if to_road == "A" else "red"

    if USE_LEDS:
        set_light(ROAD_A_PINS if to_road == "B" else ROAD_B_PINS, "yellow")

    print_traffic_state(light_a, light_b)

    t0 = time.time()
    while time.time() - t0 < YELLOW_TIME:
        _, _, cont = display_realtime_frame(
            camera_a, camera_b, model, light_a, light_b,
            0, 0, 0, None, f"YELLOW — Transitioning to Road {to_road}",
            producer=producer, cycle=cycle)
        if not cont:
            return False
        time.sleep(0.03)

    set_all_red()
    t0 = time.time()
    while time.time() - t0 < ALL_RED_TIME:
        _, _, cont = display_realtime_frame(
            camera_a, camera_b, model, "red", "red",
            0, 0, 0, None, "ALL RED — Clearing intersection",
            producer=producer, cycle=cycle)
        if not cont:
            return False
        time.sleep(0.03)

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAFFIC CONTROL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_traffic_control(camera_a, camera_b, model, producer=None):
    """
    Real-time dynamic traffic control.
    Every detection cycle publishes to Kafka:
      • vehicle-detection-events  (one event per road per frame)
      • congestion-metrics        (one event per light-switch decision)
    """
    print("\n" + "="*60)
    print("REAL-TIME DYNAMIC TRAFFIC CONTROL  [Kafka enabled]")
    print("="*60)
    print("Press 'Q' in video window or Ctrl+C to stop\n")

    cv2.namedWindow("Dynamic Traffic Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dynamic Traffic Control", 1280, 480)

    current_green      = None      # None=standby, "A" or "B"
    phase_start        = None
    current_green_time = MIN_GREEN_TIME
    cycle_count        = 0

    try:
        while True:
            frame_a = camera_a.capture_array()
            frame_b = camera_b.capture_array()

            count_a, types_a, _ = detect_and_count(model, frame_a, ROI_ROAD_A)
            count_b, types_b, _ = detect_and_count(model, frame_b, ROI_ROAD_B)
            total = count_a + count_b

            # ── STANDBY ─────────────────────────────────────────────────────
            if current_green is None:
                if total == 0:
                    if USE_LEDS:
                        set_all_red()
                    _, _, cont = display_realtime_frame(
                        camera_a, camera_b, model, "red", "red",
                        count_a, count_b, 0, None,
                        "STANDBY — No vehicles detected…",
                        producer=producer, cycle=cycle_count)
                    if not cont:
                        break
                    time.sleep(0.03)
                    continue

                # Vehicles appeared → pick first green road
                cycle_count += 1
                current_green = "B" if count_b > count_a else "A"
                green_a, green_b        = calculate_proportional_times(count_a, count_b)
                current_green_time      = green_a if current_green == "A" else green_b
                phase_start             = time.time()

                print(f"\n{'='*60}")
                print(f"CYCLE {cycle_count} — Road {current_green} GREEN for {current_green_time:.1f}s")
                print(f"  A: {count_a} vehicles | B: {count_b} vehicles")

                # ── Kafka: congestion metric on new cycle ────────────────────
                publish_congestion_metric(
                    producer, count_a, count_b,
                    current_green, current_green_time, cycle_count)

                if USE_LEDS:
                    set_light(ROAD_A_PINS, "green" if current_green == "A" else "red")
                    set_light(ROAD_B_PINS, "green" if current_green == "B" else "red")
                continue

            # ── ACTIVE ──────────────────────────────────────────────────────
            elapsed = time.time() - phase_start
            light_a = "green" if current_green == "A" else "red"
            light_b = "green" if current_green == "B" else "red"

            green_a, green_b = calculate_proportional_times(count_a, count_b)
            target = green_a if current_green == "A" else green_b
            status = (f"Road {current_green} GREEN | "
                      f"A:{count_a} B:{count_b} | "
                      f"Elapsed:{elapsed:.1f}s / Target:{target:.1f}s")

            new_a, new_b, cont = display_realtime_frame(
                camera_a, camera_b, model, light_a, light_b,
                count_a, count_b, elapsed, target, status,
                producer=producer, cycle=cycle_count)
            if not cont:
                break

            if new_a is not None:
                count_a, count_b = new_a, new_b

            # ── SWITCH DECISION ──────────────────────────────────────────────
            should_switch  = False
            switch_reason  = ""

            if elapsed < MIN_GREEN_TIME:
                pass
            elif elapsed >= MAX_WAIT_TIME:
                should_switch = True
                switch_reason = "Max wait time reached"
            elif elapsed >= target:
                should_switch = True
                switch_reason = f"Proportional time ({target:.1f}s) done"

            if should_switch:
                other = "B" if current_green == "A" else "A"
                print(f"\n🔄 SWITCH: {switch_reason}")
                print(f"   A:{count_a}  B:{count_b}")

                if total == 0:
                    print("  → No vehicles → STANDBY")
                    if not run_transition(camera_a, camera_b, model, other, producer, cycle_count):
                        break
                    current_green = None
                    continue

                green_a, green_b   = calculate_proportional_times(count_a, count_b)
                new_green_time     = green_b if other == "B" else green_a

                print(f"  → Road {other} GREEN for {new_green_time:.1f}s")

                # ── Kafka: congestion metric on switch ───────────────────────
                cycle_count += 1
                publish_congestion_metric(
                    producer, count_a, count_b,
                    other, new_green_time, cycle_count)

                if not run_transition(camera_a, camera_b, model, other, producer, cycle_count):
                    break

                current_green      = other
                current_green_time = new_green_time
                phase_start        = time.time()

                if USE_LEDS:
                    set_light(ROAD_A_PINS, "green" if current_green == "A" else "red")
                    set_light(ROAD_B_PINS, "green" if current_green == "B" else "red")

                print_traffic_state(
                    "green" if current_green == "A" else "red",
                    "green" if current_green == "B" else "red")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n\n⚠  Traffic control stopped by user")
    finally:
        print("\n🛑 Shutting down traffic control…")
        set_all_red()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("DYNAMIC TRAFFIC LIGHT CONTROL SYSTEM  (with Kafka)")
    print("="*60)
    print(f"Mode: {'LED Hardware' if USE_LEDS else 'Terminal Visualisation'}")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP}")
    print("="*60)

    setup_gpio()

    # Connect to Kafka (gracefully degrades if broker unavailable)
    producer = build_producer()

    camera_a, camera_b = initialize_cameras()
    model              = initialize_yolo()

    if camera_a is None or camera_b is None or model is None:
        print("✗ Initialisation failed — aborting")
        if producer:
            producer.close()
        return

    try:
        start_traffic = run_live_predictions(camera_a, camera_b, model, producer)

        if start_traffic:
            set_all_red()
            time.sleep(1)
            run_traffic_control(camera_a, camera_b, model, producer)

    finally:
        print("\n🛑 Shutting down…")
        set_all_red()
        if producer:
            producer.flush()   # drain in-flight messages
            producer.close()
            print("✓ Kafka producer flushed and closed")
        if camera_a:
            camera_a.stop()
        if camera_b:
            camera_b.stop()
        cleanup_gpio()
        cv2.destroyAllWindows()
        print("✓ Shutdown complete")


if __name__ == "__main__":
    main()