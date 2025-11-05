import cv2
import os
import numpy as np
import sqlite3
import time
import datetime
from flask import Flask, Response, render_template, request, jsonify, make_response
import base64
import io
from PIL import Image
from fpdf import FPDF
import odoorpc 
import datetime # Make sure datetime is imported

# --- Odoo Configuration ---
# !! THIS IS UPDATED WITH YOUR ODOO.COM DETAILS !!
ODOO_URL = 'Your Odoo.com URL'  # Your Odoo.com URL
ODOO_DB = 'Your Database Name'     # Your Database Name
ODOO_USER = 'Your Odoo user email'      # Your Odoo user email
# --- IMPORTANT ---
# PASTE YOUR NEW ODOO API KEY HERE. Your regular password will not work
# because you have 2-Factor Authentication enabled.
ODOO_PASSWORD = 'API KEY'           

# --- Configuration ---
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
TRAINER_FILE = 'trainer.yml'
DATABASE_FILE = 'attendance.db' # Still used for user name mapping
DATASET_PATH = 'dataset'
CONFIDENCE_THRESHOLD = 50 

# --- Initialize Flask App ---
# NOTE: We specify the 'templates' folder for HTML files
app = Flask(__name__, template_folder='templates')

# --- Helper: Create dataset directory ---
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# --- Global Variables for Detection ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

# --- Global State for API ---
app.last_action_time = 0
COOLDOWN_SECONDS = 5
user_map = {} # Will be loaded at startup

# --- Database Helper Functions (Still used for training) ---
# (Your original init_db, load_user_map_from_db, etc. remain unchanged)

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'checked-out' 
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database '{DATABASE_FILE}' initialized.")

def load_user_map_from_db():
    """Loads the user ID-to-name mapping from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id, name FROM users")
        rows = cursor.fetchall()
        return {str(row[0]): row[1] for row in rows}
    except sqlite3.Error as e:
        print(f"Error loading user map from DB: {e}")
        return {}
    finally:
        conn.close()

def get_user_by_name_from_db(name):
    """Finds a user by name (case-insensitive) and returns their ID."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT user_id FROM users WHERE LOWER(name) = LOWER(?)", (name,))
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"Error fetching user by name: {e}")
        return None
    finally:
        conn.close()

def get_next_user_id_from_db():
    """Finds the highest user ID and returns the next available ID."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT MAX(user_id) FROM users")
        row = cursor.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return max_id + 1
    except sqlite3.Error as e:
        print(f"Error fetching next user ID: {e}")
        return 1
    finally:
        conn.close()

def create_user_in_db(user_id, name):
    """Saves a new user to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (user_id, name) VALUES (?, ?)", (user_id, name))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Error: User name '{name}' or ID '{user_id}' already exists.")
        return False
    except sqlite3.Error as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        conn.close()
    return True

# --- NEW: Odoo Attendance Function ---
def record_attendance_odoo(user_name):
    """
    Connects to Odoo and triggers an attendance action for the employee.
    The employee name in Odoo MUST match the 'user_name' perfectly.
    
    *** NEW LOGIC ***
    This function now replicates the logic from Odoo's private
    '_attendance_action_change' method to be version-independent.
    """
    try:
        print(f"[Odoo] Attempting to connect to {ODOO_URL}...")

        # *** THIS IS UPDATED FOR ODOO.COM ***
        # We connect to your Odoo.com host using the 'jsonrpc+ssl' protocol on port 443
        odoo = odoorpc.ODOO(
            ODOO_URL.replace('https://', ''), 
            protocol='jsonrpc+ssl', 
            port=443
        )
        odoo.login(ODOO_DB, ODOO_USER, ODOO_PASSWORD)

        print(f"[Odoo] Connection successful.")

        # 1. Find the Odoo Employee
        Employee = odoo.env['hr.employee']
        employee_ids = Employee.search([('name', '=', user_name)])
        
        if not employee_ids:
            msg = f"Odoo Error: Employee '{user_name}' not found."
            print(f"[Odoo] {msg}")
            return (msg, False)
            
        employee_id = employee_ids[0]
        # Get a "browsable" record to read its fields
        employee = Employee.browse(employee_id)
        
        print(f"[Odoo] Found employee: {employee.name} (ID: {employee.id})")

        # 2. Read the employee's current attendance state
        # This field ('attendance_state') is confirmed from your hr_employee.py file
        current_state = employee.attendance_state
        print(f"[Odoo] Employee current state: {current_state}")

        Attendance = odoo.env['hr.attendance']
        action_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # 3. Perform the correct action based on the state
        if current_state == 'checked_out':
            # ACTION: Check In
            # We create a new attendance record
            print(f"[Odoo] Action: Checking IN")
            vals = {
                'employee_id': employee_id,
                'check_in': action_date,
            }
            new_att_id = Attendance.create(vals)
            print(f"[Odoo] Created new attendance record (ID: {new_att_id})")
            message = f"'{user_name}' checked in successfully in Odoo."
            return (message, True)

        elif current_state == 'checked_in':
            # ACTION: Check Out
            # We find the last "open" attendance record and close it
            print(f"[Odoo] Action: Checking OUT")
            
            # Find attendance record for this employee that has no check_out
            domain = [
                ('employee_id', '=', employee_id),
                ('check_out', '=', False)
            ]
            attendance_ids = Attendance.search(domain, limit=1)
            
            if not attendance_ids:
                msg = f"Odoo Error: Cannot check out. No open check-in record found for '{user_name}'."
                print(f"[Odoo] {msg}")
                return (msg, False)

            # Get the one record to update
            attendance_to_close = Attendance.browse(attendance_ids[0])
            attendance_to_close.write({'check_out': action_date})
            
            print(f"[Odoo] Closed attendance record (ID: {attendance_ids[0]})")
            message = f"'{user_name}' checked out successfully in Odoo."
            return (message, True)

        else:
            msg = f"Odoo Error: Unknown attendance state '{current_state}'."
            print(f"[Odoo] {msg}")
            return (msg, False)

    except Exception as e:
        msg = f"Odoo API Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return (msg, False)


# --- Image Decoding Function (Unchanged) ---
def decode_base64_image(base64_string):
    """Decodes a base64 string into a CV2-compatible image."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        img_bytes = base64.b64decode(base64_string)
        pil_img = Image.open(io.BytesIO(img_bytes))
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return open_cv_image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Training Helper (Unchanged) ---
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        if not image_path.endswith('.jpg'):
            continue
        try:
            pil_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if pil_img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            
            img_numpy = np.array(pil_img, 'uint8')
            id_str = os.path.split(image_path)[-1].split(".")[1]
            id = int(id_str)
            
            ids.append(id)
            face_samples.append(img_numpy)
        except Exception as e:
            print(f"Warning: Skipping file {image_path}. Error: {e}")
            
    return face_samples, ids

# --- Helper to load recognizer (Unchanged) ---
def load_recognizer():
    global recognizer, user_map
    if not os.path.exists(TRAINER_FILE):
        print(f"Warning: '{TRAINER_FILE}' not found. Training needed.")
        return
    try:
        recognizer.read(TRAINER_FILE)
        user_map = load_user_map_from_db()
        print(f"[INFO] Models reloaded. {len(user_map)} user(s) found.")
    except cv2.error as e:
        print(f"Error loading recognizer: {e}. Is trainer.yml valid?")

# --- Format Duration Helper (Unchanged) ---
def format_duration(seconds):
    # (Your original function)
    if seconds is None or seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0m 0s"
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


# --- Flask API Routes ---

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """Receives a frame from the client, recognizes it, and returns the user info."""
    # (This function is unchanged)
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"status": "error", "message": "No image data provided"}), 400

    img = decode_base64_image(data['image'])
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image data"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"status": "no_face"})

    (x, y, w, h) = faces[0]
    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    
    if confidence < CONFIDENCE_THRESHOLD:
        name = user_map.get(str(id), "Unknown ID")
        return jsonify({
            "status": "known", 
            "user_id": id, 
            "user_name": name,
            "confidence": round(100 - confidence)
        })
    else:
        return jsonify({"status": "unknown", "confidence": round(100 - confidence)})

# --- MODIFIED Attendance Route ---
@app.route('/attendance_action', methods=['POST'])
def attendance_action():
    """
    Handles check-in and check-out button presses.
    MODIFIED: This now calls the Odoo API function.
    """
    now = time.time()
    if now - app.last_action_time < COOLDOWN_SECONDS:
        return jsonify({"success": False, "message": "Please wait..."})
    
    data = request.json
    if data.get('user_id') is None or data.get('user_name') is None:
        return jsonify({"success": False, "message": "No known face detected!"})

    # --- THIS IS THE CHANGE ---
    user_name = data.get('user_name') # Get the name from the request
    
    # OLD WAY:
    # message, is_success = record_attendance_db(data['user_id'], data['user_name'], data['action']) 
    
    # NEW WAY:
    # We pass the 'user_name'. Odoo handles the 'action' (in/out) logic.
    message, is_success = record_attendance_odoo(user_name)
    # --- END OF CHANGE ---

    if is_success:
        app.last_action_time = now
    
    return jsonify({"success": is_success, "message": message})


# --- NEW TRAINING ROUTES (Unchanged) ---
# (Your original /train, /start_training, /capture_frame, 
# and /run_model_training routes remain unchanged)

@app.route('/train')
def train_page():
    """Serves the new training webpage."""
    return render_template('train.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    """Creates a new user (or finds existing) and returns the user ID."""
    data = request.json
    user_name = data.get('name', '').strip()
    if not user_name:
        return jsonify({"success": False, "message": "Name cannot be empty."})

    user_id = get_user_by_name_from_db(user_name)
    if user_id is not None:
        print(f"User '{user_name}' (ID: {user_id}) already exists. Adding more images.")
        # Clear old images for this user
        for f in os.listdir(DATASET_PATH):
            if f.startswith(f"User.{user_id}."):
                os.remove(os.path.join(DATASET_PATH, f))
        print(f"Removed old images for user ID {user_id}.")
    else:
        user_id = get_next_user_id_from_db()
        if not create_user_in_db(user_id, user_name):
            return jsonify({"success": False, "message": "Error creating user in DB."})
        print(f"Creating new user '{user_name}' with ID {user_id}.")
        
    return jsonify({"success": True, "user_id": user_id, "user_name": user_name})

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Receives and saves a single frame for training."""
    data = request.json
    user_id = data.get('user_id')
    count = data.get('count')
    image_data = data.get('image')

    if not all([user_id, count, image_data]):
        return jsonify({"success": False, "message": "Missing data."})

    img = decode_base64_image(image_data)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image data."})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "No face detected in frame."})
    
    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    
    img_path = os.path.join(DATASET_PATH, f"User.{user_id}.{count}.jpg")
    cv2.imwrite(img_path, face_img)
    
    return jsonify({"success": True, "message": f"Saved {img_path}"})

@app.route('/run_model_training', methods=['POST'])
def run_model_training():
    """Triggers the backend to train the LBPH model."""
    print("[INFO] Training model on all captured images...")
    
    try:
        faces, ids = get_images_and_labels(DATASET_PATH)
        if not faces:
            print("Error: No faces found to train.")
            return jsonify({"success": False, "message": "No faces found in dataset."})

        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAINER_FILE)
        
        # Reload the recognizer and user map in memory
        load_recognizer()
        
        print(f"\n[INFO] {len(np.unique(ids))} user(s) trained. Model saved to {TRAINER_FILE}.")
        return jsonify({"success": True, "message": "Training complete! Model updated."})

    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"success": False, "message": f"Error during training: {e}"})

# --- PDF Helper Class (Unchanged) ---
# (Your original PDF class)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Attendance Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def report_table(self, data, col_widths):
        self.set_font('Arial', 'B', 10)
        headers = ['Name', 'Date', 'First In', 'Last Out', 'Total Duration', 'All Sessions']
        col_keys = ['name', 'date', 'first_in', 'last_out', 'duration', 'sessions']
        for i, header in enumerate(headers):
            self.cell(col_widths[col_keys[i]], 10, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        if not data:
            self.cell(sum(col_widths.values()), 10, 'No data found for this selection.', 1, 1, 'C')
            return
        for row in data:
            start_y = self.get_y()
            row_data_points = [
                str(row['name']),
                str(row['event_date']),
                str(row['first_check_in']),
                str(row['last_check_out']),
                str(row['total_duration_formatted']),
                str(row['all_sessions'])
            ]
            cell_heights = []
            current_x = self.get_x()
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                self.multi_cell(width, 8, row_data_points[i], 0, 'L')
                cell_heights.append(self.get_y())
                current_x += width
                self.set_xy(current_x, start_y)
            max_y = max(cell_heights)
            current_x = self.l_margin
            self.set_xy(current_x, start_y)
            for i, key in enumerate(col_keys):
                width = col_widths[key]
                self.rect(current_x, start_y, width, max_y - start_y)
                current_x += width
            self.set_xy(self.l_margin, max_y)

# --- Internal Report Data Helper (Unchanged) ---
# (Your original _get_report_data function)
# This will now only report on the local DB, which is no longer used
# for attendance. All new reports should be viewed inside Odoo.
def _get_report_data(user_id, start_date, end_date):
    """
    Internal function to query ODOO for a smart summary report.
    This replaces the old function that read from SQLite.
    """
    try:
        print("[Odoo] Connecting for report...")
        odoo = odoorpc.ODOO(
            ODOO_URL.replace('https://', ''), 
            protocol='jsonrpc+ssl', 
            port=443
        )
        odoo.login(ODOO_DB, ODOO_USER, ODOO_PASSWORD)
        print("[Odoo] Report connection successful.")
        
        Attendance = odoo.env['hr.attendance']
        domain = []
        
        if user_id:
            domain.append(('employee_id', '=', int(user_id)))
        if start_date:
            # Odoo's datetime fields are UTC.
            domain.append(('check_in', '>=', f"{start_date} 00:00:00"))
        if end_date:
            domain.append(('check_in', '<=', f"{end_date} 23:59:59"))

        # Fetch all relevant attendance records
        # odoorpc < 0.8 returns datetimes as strings. >= 0.8 returns datetime objects
        # Let's force string conversion for safety and parse manually.
        attendances = Attendance.search_read(
            domain,
            fields=['employee_id', 'check_in', 'check_out', 'worked_hours'],
            order='check_in asc'
        )
        
        # Process data in Python
        report_data = {} # Key: (employee_id, date_str)
        
        for att in attendances:
            # att['employee_id'] is a list: [id, name]
            employee_id = att['employee_id'][0]
            employee_name = att['employee_id'][1]
            
            check_in_str = att['check_in']
            if not check_in_str:
                continue # Should not happen, but good to check
            
            # Parse naive datetime string from Odoo (which is in UTC)
            check_in_dt = datetime.datetime.strptime(check_in_str, '%Y-%m-%d %H:%M:%S')
            
            event_date_str = check_in_dt.strftime('%Y-%m-%d')
            key = (employee_id, event_date_str)

            # Initialize dict for this employee on this day
            if key not in report_data:
                report_data[key] = {
                    'name': employee_name,
                    'event_date': event_date_str,
                    'check_ins': [],
                    'check_outs': [],
                    'sessions': [],
                    'total_duration_seconds': 0
                }
            
            report_data[key]['check_ins'].append(check_in_dt)
            
            # Add duration. Odoo 'worked_hours' is in hours (float)
            report_data[key]['total_duration_seconds'] += (att['worked_hours'] * 3600)
            
            check_out_str = att['check_out'] # This can be False or a string
            
            if check_out_str:
                check_out_dt = datetime.datetime.strptime(check_out_str, '%Y-%m-%d %H:%M:%S')
                report_data[key]['check_outs'].append(check_out_dt)
                session_str = f"{check_in_dt.strftime('%H:%M:%S')} - {check_out_dt.strftime('%H:%M:%S')}"
            else:
                session_str = f"{check_in_dt.strftime('%H:%M:%S')} - (Still In)"
            
            report_data[key]['sessions'].append(session_str)

        # Finalize the list for the report
        final_list = []
        for key, data in sorted(report_data.items(), key=lambda item: (item[1]['event_date'], item[1]['name']), reverse=True):
            first_in = min(data['check_ins']).strftime('%H:%M:%S') if data['check_ins'] else "---"
            last_out = max(data['check_outs']).strftime('%H:%M:%S') if data['check_outs'] else "---"
            all_sessions = "\n".join(data['sessions']) if data['sessions'] else "No sessions"
            
            final_list.append({
                'name': data['name'],
                'event_date': data['event_date'],
                'first_check_in': first_in,
                'last_check_out': last_out,
                'total_duration_seconds': data['total_duration_seconds'],
                'total_duration_formatted': format_duration(data['total_duration_seconds']),
                'all_sessions': all_sessions
            })
        
        return final_list, None

    except Exception as e:
        msg = f"Odoo Report Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return None, msg


# --- REPORTING & ADMIN ROUTES (Unchanged) ---
# (These routes remain unchanged, but will only report on the local DB)
@app.route('/reports')
def reports_page():
    """Serves the new reports/admin page."""
    return render_template('reports.html')

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Fetches all users from ODOO for the report filter dropdown."""
    try:
        print("[Odoo] Connecting for user list...")
        odoo = odoorpc.ODOO(
            ODOO_URL.replace('https://', ''), 
            protocol='jsonrpc+ssl', 
            port=443
        )
        odoo.login(ODOO_DB, ODOO_USER, ODOO_PASSWORD)
        
        Employee = odoo.env['hr.employee']
        # Fetch all employees, just id and name, order by name
        employees = Employee.search_read([], fields=['id', 'name'], order='name asc')
        
        # Format as the UI expects
        user_list = [{"user_id": emp['id'], "name": emp['name']} for emp in employees]
        print(f"[Odoo] Found {len(user_list)} employees.")
        return jsonify(user_list)
        
    except Exception as e:
        msg = f"Odoo User Fetch Error: {str(e)}"
        print(f"[Odoo] {msg}")
        return jsonify({"error": msg}), 500

@app.route('/api/attendance_report', methods=['GET'])
def get_attendance_report_json():
    """Provides the attendance report data as JSON for the web table."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data, error = _get_report_data(user_id, start_date, end_date)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(data)

@app.route('/api/download_pdf', methods=['GET'])
def download_pdf_report():
    """Generates and serves the attendance report as a PDF download."""
    user_id = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    data, error = _get_report_data(user_id, start_date, end_date)
    if error:
        return f"Error generating report: {error}", 500
    pdf = PDF(orientation='L', format='A4')
    pdf.add_page()
    col_widths = {'name': 45, 'date': 25, 'first_in': 25, 'last_out': 25, 'duration': 30, 'sessions': 127}
    pdf.report_table(data, col_widths)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_report_{timestamp}.pdf"
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


# --- Main ---
if __name__ == "__main__":
    init_db() # Ensure local DB exists for training
    load_recognizer() # Load models on startup
    
    print("[INFO] Starting Flask server...")
    print("[INFO] To access from Odoo/phone, use https://<YOUR_PC_IP_ADDRESS>:5000")
    
    # Use adhoc SSL context for HTTPS, which is required for
    # cameras in modern browsers and for embedding in an HTTPS Odoo.
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, 
            use_reloader=False, ssl_context='adhoc')