import cv2
import numpy as np
import json

# Load the taught colors
with open("taught_colors.json", "r") as f:
    taught_colors = json.load(f)

# Matrices to store detected colors
initial_cube_matrix = [None, None, None]
final_cube_matrix = [None, None, None]

# Define the regions for each row (y-coordinate ranges)
row_regions = [
    (0, 160),      # Row 1: y from 0 to 160
    (160, 320),    # Row 2: y from 160 to 320
    (320, 480)     # Row 3: y from 320 to 480
]

def detect_cube_position(position_type):
    """
    Detect cube colors for either initial or final position
    
    Args:
        position_type: String, either "INITIAL" or "FINAL"
    
    Returns:
        List of detected colors
    """
    global initial_cube_matrix, final_cube_matrix
    
    # Select the appropriate matrix based on position type
    if position_type == "INITIAL":
        cube_matrix = initial_cube_matrix
        json_filename = "cube_matrix.json"
    else:  # FINAL
        cube_matrix = final_cube_matrix
        json_filename = "final_cube_matrix.json"
        
    current_row = 0
    total_rows = 3
    processing_complete = False
    
    cap = cv2.VideoCapture(0)
    
    print(f"{position_type} POSITION: Starting with Row 1. Press ENTER when the row is complete.")
    
    while not processing_complete:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Highlight the current row
        row_start, row_end = row_regions[current_row]
        highlight_frame = frame.copy()
       
        overlay = highlight_frame.copy()
        cv2.rectangle(overlay, (0, row_start), (640, row_end), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, highlight_frame, 0.7, 0, highlight_frame)
        
        # Draw horizontal lines to show row boundaries
        for y in [160, 320]:
            cv2.line(highlight_frame, (0, y), (640, y), (255, 255, 255), 1)
        
        # Extract the current row region
        row_start, row_end = row_regions[current_row]
        roi = frame[row_start:row_end, :]
        hsv_roi = hsv[row_start:row_end, :]
        
        # Color detection for current row
        detected_color = None
        for color_name, ranges in taught_colors.items():
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            mask = cv2.inRange(hsv_roi, lower, upper)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Adjust y to global coordinates for display
                    y_global = y + row_start
                    
                    cv2.rectangle(highlight_frame, (x, y_global), (x + w, y_global + h), (0, 255, 0), 2)
                    cv2.putText(highlight_frame, color_name, (x, y_global - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Store the detected color
                    detected_color = color_name
        
        # Update the cube matrix with detected color
        if detected_color:
            cube_matrix[current_row] = detected_color
        
        # Display processing status
        cv2.putText(highlight_frame, f"{position_type} POSITION: Processing Row {current_row + 1}/3", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detected_color:
            cv2.putText(highlight_frame, f"Detected: {detected_color}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(highlight_frame, "No color detected", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current matrix state
        matrix_str = f"{position_type} Matrix: ["
        for i, color in enumerate(cube_matrix):
            if i == current_row:
                # Highlight current row
                matrix_str += f">{color}<" if color else ">None<"
            else:
                matrix_str += f"{color}" if color else "None"
            
            if i < len(cube_matrix) - 1:
                matrix_str += ", "
        matrix_str += "]"
        
        cv2.putText(highlight_frame, matrix_str, 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(highlight_frame, "Press ENTER to confirm row and proceed", 
                   (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(highlight_frame, "Press 'q' to quit", 
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show the frame
        cv2.imshow(f"{position_type} Position - Color Detection", highlight_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 13:  # Enter key
            # Move to the next row
            print(f"{position_type} Position Row {current_row + 1} completed with color: {cube_matrix[current_row]}")
            current_row += 1
            
            if current_row >= total_rows:
                print(f"All {position_type} rows processed!")
                print(f"{position_type} cube matrix: {cube_matrix}")
                processing_complete = True
            else:
                print(f"Moving to {position_type} Position Row {current_row + 1}. Press ENTER when the row is complete.")
    
    # Save the matrix to a JSON file
    with open(json_filename, "w") as f:
        json.dump(cube_matrix, f)
    print(f"{position_type} cube matrix saved to {json_filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the detected matrix
    return cube_matrix

# Main execution
print("Starting cube position detection process")
print("First, we'll detect the INITIAL position of the cube")
initial_matrix = detect_cube_position("INITIAL")

print("\n-----------------------------------------------------")
print("Now, we'll detect the FINAL position of the cube")
print("Please position your cube in its final state.")
input("Press ENTER when ready to begin final position detection...")
final_matrix = detect_cube_position("FINAL")

print("\n-----------------------------------------------------")
print("Detection complete!")
print(f"Initial cube matrix: {initial_matrix}")
print(f"Final cube matrix: {final_matrix}")
print("The data has been saved to 'cube_matrix.json' and 'final_cube_matrix.json'")



