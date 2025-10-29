# Programmed in the wake of John Thummel. Rest In Peace you crazy genius.

import cv2
import numpy as np
import time

class CurtainSystem:
    def __init__(self, width, height, spacing=60):
        # Create arrays for vectorized operations
        x_coords = np.arange(0, width, spacing, dtype=np.float32)
        self.num_points = len(x_coords)
        
        # Arrays for vectorized physics
        self.x = x_coords
        self.y = np.full(self.num_points, height, dtype=np.float32)
        self.velocity = np.zeros(self.num_points, dtype=np.float32)
        self.target_y = self.y.copy()
        self.rest_y = self.y.copy()
        
        # Pre-compute line coordinates for faster drawing
        self.line_indices = np.column_stack((np.arange(self.num_points-1), np.arange(1, self.num_points)))
        
        # Store dimensions for buffer creation
        self.width = width
        self.height = height
        
    def update(self, dt, gravity=980.0, damping=0.8):
        # Vectorized physics update
        above_rest = self.y < self.rest_y
        self.velocity[above_rest] += gravity * dt
        
        # Apply damping
        self.velocity *= (1.0 - damping * dt)
        
        # Update positions
        self.y += self.velocity * dt
        
        # Handle target positions
        below_target = self.y > self.target_y
        self.y[below_target] = self.target_y[below_target]
        self.velocity[below_target] = 0
        
        # Clamp to rest position
        above_rest = self.y > self.rest_y
        self.y[above_rest] = self.rest_y[above_rest]
        self.velocity[above_rest] = 0
    
    def draw(self, frame, thickness=8):
        height, width = frame.shape[:2]
        # Create drawing buffer matching frame dimensions
        draw_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert coordinates to integers for drawing
        points = np.column_stack((self.x.astype(np.int32), self.y.astype(np.int32)))
        
        # Ensure points are within frame boundaries
        points = np.clip(points, [0, 0], [width - 1, height - 1])
        
        # Draw lines using vectorized operations
        for i in range(len(points) - 1):
            cv2.line(draw_buffer, tuple(points[i]), tuple(points[i+1]), (255, 255, 255), thickness)
        
        # Apply the curtain to the frame
        mask = cv2.inRange(draw_buffer, (255, 255, 255), (255, 255, 255))
        frame[mask > 0] = (255, 255, 255)

def load_yolo():
    # Load YOLOv3-tiny for faster inference
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # Enable OpenCV optimizations
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers

def detect_people(frame, net, output_layers, curtain_system, last_time, maskOnly=False, show_fps=False):
    current_time = time.time()
    dt = current_time - last_time
    dt = min(dt, 0.1)  # Clamp dt to prevent large jumps
    
    # Start timing the processing
    process_start_time = time.time()
    
    # Reduce resolution for faster processing
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    height, width = small_frame.shape[:2]
    
    # Prepare image for YOLO model (smaller input size for speed)
    blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    # We'll use curtain_points passed as parameter instead of creating new points

    # Information for detection
    boxes = []
    confidences = []
    class_ids = []

    # Process detections using numpy operations
    all_detections = np.vstack(outs)
    
    # Filter for person class (class 0) and confidence threshold
    scores = all_detections[:, 5:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    mask = (class_ids == 0) & (confidences > 0.5)
    
    filtered_detections = all_detections[mask]
    
    if len(filtered_detections) > 0:
        # Convert to original image coordinates
        boxes = filtered_detections[:, :4]
        boxes[:, [0, 2]] *= width / scale  # Adjust for reduced frame size
        boxes[:, [1, 3]] *= height / scale
        
        # Convert to x, y, w, h format
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y
        boxes_xywh[:, 2] = boxes[:, 2]  # width
        boxes_xywh[:, 3] = boxes[:, 3]  # height
        
        # Apply non-max suppression
        filtered_confidences = confidences[mask]
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), filtered_confidences.tolist(), 0.5, 0.4)
        
        if len(indices) > 0:
            # Reset target positions
            curtain_system.target_y[:] = curtain_system.rest_y
    
            # Process each detected box using vectorized operations
            boxes_xywh = boxes_xywh[indices.flatten()]
            
            for box in boxes_xywh:
                x, y, w, h = map(int, box)
                center_x = x + w//2
                fall_off_factor = 0.5
                
                # Find points within the box using vectorized operations
                mask = (curtain_system.x >= x) & (curtain_system.x <= x + w)
                if not np.any(mask):
                    continue
                
                # Find closest point to center
                points_in_box = curtain_system.x[mask]
                distances = np.abs(points_in_box - center_x)
                closest_idx = np.argmin(distances)
                
                # Calculate distance factors for all points in box
                dist_factors = np.abs(points_in_box - center_x) / (w/2)
                height_factors = 1.0 - (fall_off_factor * np.square(dist_factors))
                height_factors = np.power(height_factors, 2.5)
                
                # Update target positions using vectorized operations
                target_y = y + (curtain_system.rest_y[mask] - y) * (1 - height_factors)
                curtain_system.target_y[mask] = target_y
    
    # Update physics with vectorized operations
    curtain_system.update(dt)
    
    # Draw curtain
    curtain_system.draw(frame)
    
    # Create a mask for pure white pixels (255, 255, 255)
    mask = cv2.inRange(frame, (255, 255, 255), (255, 255, 255))
    
    # Calculate processing time
    process_time = (time.time() - process_start_time) * 1000  # Convert to milliseconds
    
    if show_fps:
        # Add processing time text in the upper right corner
        fps = 1000 / process_time if process_time > 0 else 0
        text = f"{process_time:.1f}ms ({fps:.1f}FPS)"
        # Get text size for positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position in upper right with padding
        padding = 10
        text_x = width - text_width - padding
        text_y = text_height + padding
        
        # Draw black background for better visibility
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, (0, 255, 0), thickness)
    
    if maskOnly:
        # Create the final output frame (black background with only pure white pixels)
        final_frame = np.zeros_like(frame)
        final_frame[mask > 0] = (255, 255, 255)
        return final_frame, current_time
    else:
        return frame, current_time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the camera was opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Available cameras:")
        # Try other camera indices
        for i in range(4):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"Camera {i} is available")
            test_cap.release()
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized successfully:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Try to get the first frame with retry
    max_retries = 5
    frame = None
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        print(f"Attempt {attempt + 1}: Failed to read frame, retrying...")
        time.sleep(1)
    
    if frame is None:
        print("Error: Could not read any frames from the camera")
        cap.release()
        return
        
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Initialize optimized curtain system
    spacing = int(width / 60)
    bottom_offset = 5
    maskOnly = False  # Set to True to output only the mask of white pixels
    show_fps = False  # Set to True to display processing time and FPS
    curtain_system = CurtainSystem(width, height - bottom_offset, spacing)
    
    # Initialize time
    last_time = time.time()
    
    try:
        # Load YOLO model
        net, output_layers = load_yolo()
        
        while True:
            _, frame = cap.read()
            if frame is None:
                break
                
            # Detect and draw
            frame, last_time = detect_people(frame, net, output_layers, curtain_system, last_time, maskOnly, show_fps)
            
            # Resize frame to desired dimensions
            # frame = cv2.resize(frame, (output_width, output_height))
            
            # Display the result
            cv2.imshow("People Detection", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
