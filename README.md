# 🎨 Color Detection + Teaching for Dobot Magician

This repository contains an experimental **computer vision + robotics** pipeline that:

- Detects **colors** using a webcam and OpenCV  
- Lets you **“teach”** the system which color belongs to which **position / cube coordinate**  
- Syncs the detected color with **Dobot Magician** coordinates so the robot can interact with the correct object

The goal is to create a simple **teaching-by-demonstration** workflow for color-based object handling.

---

## 🧠 Concept Overview

1. **Teach Phase**  
   You show the system a color (e.g., a cube on the workspace), and assign / confirm its position.  
   The script stores this mapping (color → coordinate / cube index) in JSON files.

2. **Detect Phase**  
   The camera detects a color in real time.  
   Using the taught mappings, the system looks up the corresponding target coordinate and can then drive the **Dobot Magician** to that location.

This is useful for:
- Color-sorting demos  
- Teaching robots to pick up colored cubes  
- Simple HRI demos in robotics labs or workshops  

---


## 📁 Repository Structure

<table>
<tr>
<th>📄 File</th>
<th>🧠 Purpose</th>
<th>🤖 Usage</th>
</tr>
<tr>
<td><code>detect_color.py</code></td>
<td>Real-time color detection & lookup using OpenCV</td>
<td>Retrieves the corresponding coordinate and triggers robot movement</td>
</tr>
<tr>
<td><code>teach_color.py</code></td>
<td>Assigns coordinates to detected colors</td>
<td>Updates JSON mappings for future autonomous execution</td>
</tr>
<tr>
<td><code>cube_matrix.json</code></td>
<td>Initial coordinate grid</td>
<td>Reference workspace layout before teaching</td>
</tr>
<tr>
<td><code>final_cube_matrix.json</code></td>
<td>Updated coordinate grid</td>
<td>Used after teaching for accurate positioning</td>
</tr>
<tr>
<td><code>taught_colors.json</code></td>
<td>Stored mapping of <b>color → coordinate / cube index</b></td>
<td>Main lookup dataset for the detection script</td>
</tr>
</table>

---

## ▶️ How to Run

### 🔧 Install Requirements
```bash
pip install opencv-python numpy
