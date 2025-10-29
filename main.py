# Programmed in the wake of John Thummel. Rest In Peace you crazy genius.

import cv2
import numpy as np
import time

class CurtainPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0.0  # Vertical velocity
        self.target_y = y    # Target height
        self.rest_y = y      # Resting position

    def update(self, dt, gravity=980.0, damping=0.8):
        # Calculate force based on distance to target
        distance_to_target = self.target_y - self.y
        
        # Apply acceleration due to gravity when above rest position
        if self.y < self.rest_y:
            self.velocity += gravity * dt
        
        # Apply damping to velocity
        self.velocity *= (1.0 - damping * dt)
        
        # Update position
        if distance_to_target < 0:
            self.velocity = 0
            self.y = self.target_y
        else:
            self.y += self.velocity * dt
        
        # Clamp position to rest_y
        if self.y > self.rest_y:
            self.y = self.rest_y
            self.velocity = 0

def load_yolo():
    # Load YOLO model using OpenCV's DNN module
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_people(frame, net, output_layers, curtain_points, last_time):
    current_time = time.time()
    dt = current_time - last_time
    dt = min(dt, 0.1)  # Clamp dt to prevent large jumps
    
    height, width, _ = frame.shape
    # Prepare image for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    # We'll use curtain_points passed as parameter instead of creating new points

    # Information for detection
    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # We only want person detections (class 0 in COCO dataset)
            if class_id == 0 and confidence > 0.5:
                # Get box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Reset all target positions to rest position
    for point in curtain_points:
        point.target_y = point.rest_y
    
    # Update target positions based on detections
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center_x = x + w//2
            fall_off_factor = 0.5  # Adjust this to control how quickly height falls off (0-1)
            
            # First find the closest point to center
            closest_point = None
            min_dist = float('inf')
            closest_idx = -1
            
            for idx, point in enumerate(curtain_points):
                if x <= point.x <= x + w:
                    dist = abs(point.x - center_x)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = point
                        closest_idx = idx
            
            # If we found a center point, update all points within the box
            if closest_point is not None:
                closest_point.target_y = y  # Center point stays at detection height
                
                # Update points to the left of center
                for idx in range(closest_idx - 1, -1, -1):
                    point = curtain_points[idx]
                    if x <= point.x <= x + w:
                        # Calculate distance from center as a percentage of half box width
                        dist_factor = abs(point.x - center_x) / (w/2)
                        # Apply quadratic fall-off for sharper drop near edges
                        # Square the distance factor and adjust curve steepness
                        height_factor = 1.0 - (fall_off_factor * (dist_factor * dist_factor))
                        # Apply exponential easing for even smoother transition near center
                        height_factor = pow(height_factor, 2.5)  # Adjust power for different curve shapes
                        point.target_y = y + (point.rest_y - y) * (1 - height_factor)
                
                # Update points to the right of center
                for idx in range(closest_idx + 1, len(curtain_points)):
                    point = curtain_points[idx]
                    if x <= point.x <= x + w:
                        # Calculate distance from center as a percentage of half box width
                        dist_factor = abs(point.x - center_x) / (w/2)
                        # Apply quadratic fall-off for sharper drop near edges
                        # Square the distance factor and adjust curve steepness
                        height_factor = 1.0 - (fall_off_factor * (dist_factor * dist_factor))
                        # Apply exponential easing for even smoother transition near center
                        height_factor = pow(height_factor, 2.5)  # Adjust power for different curve shapes
                        point.target_y = y + (point.rest_y - y) * (1 - height_factor)
    
    # Update physics for all points
    for point in curtain_points:
        point.update(dt)
    
    # Draw the curtain effect
    # First, draw the main line segments between points
    for i in range(len(curtain_points) - 1):
        pt1 = (int(curtain_points[i].x), int(curtain_points[i].y))
        pt2 = (int(curtain_points[i + 1].x), int(curtain_points[i + 1].y))
        cv2.line(frame, pt1, pt2, (255, 255, 255), 8)
    
    # # Draw the vertical "hanging" lines
    # for i in range(0, len(curtain_points), 5):
    #     pt = (int(curtain_points[i].x), int(curtain_points[i].y))
    #     cv2.line(frame, pt, (pt[0], pt[1] - 20), (255, 255, 255), 1)
    
    return frame, current_time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set desired output dimensions
    # output_width = 1920  # Adjust these values as needed
    # output_height = 1080  # Adjust these values as needed
    
    # Set camera properties (may not work with all cameras)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_height)
    
    # Get initial frame dimensions
    _, frame = cap.read()
    
    # Resize frame to desired dimensions
    # frame = cv2.resize(frame, (output_width, output_height))
    
    height, width = frame.shape[:2]
    
    # Initialize curtain points
    spacing = int(width / 60)
    bottom_offset = 5
    curtain_points = [CurtainPoint(x, height - bottom_offset) for x in range(0, width, spacing)]
    
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
            frame, last_time = detect_people(frame, net, output_layers, curtain_points, last_time)
            
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
