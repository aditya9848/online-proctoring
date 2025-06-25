from flask import Flask, render_template, request, redirect, session, url_for, Response
import json, cv2, os
from proctor import detect_cheating
from questions import tests

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

USERS = json.load(open("users.json"))
camera = cv2.VideoCapture(0)

cheating_warnings_per_test = {}
cheating_messages = {}
active_tests = {}  # key: student_test, value: {"student_id", "test_id"}

# ---- Auth routes unchanged ----

@app.route('/login', methods=['GET','POST'])
@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        sid = request.form['student_id']
        pw = request.form['password']
        if sid in USERS and USERS[sid] == pw:
            session['student_id'] = sid
            return redirect(url_for('dashboard'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---- Dashboard ----

@app.route('/dashboard')
def dashboard():
    if 'student_id' not in session:
        return redirect(url_for('login'))
    sid = session['student_id']
    if sid == 'admin':
        return redirect(url_for('admin'))
    return render_template("dashboard.html", student_id=sid)

# ---- Start test ----

@app.route('/test/<int:test_id>')
def start_test(test_id):
    sid = session.get('student_id')
    if not sid or sid == 'admin':
        return redirect(url_for('login'))

    key = f"{sid}_test{test_id}"

    # prevent reattempt
    if os.path.exists("results.json"):
        results = json.load(open("results.json"))
        if key in results:
            return redirect(url_for('result', test_id=test_id))

    cheating_warnings_per_test[key] = 0
    cheating_messages[key] = "All clear ✅"
    active_tests[key] = {"student_id": sid, "test_id": test_id}

    return render_template(
        "test.html", test_id=test_id, student_id=sid,
        questions=tests.get(test_id, [])
    )

# ---- Core stream and alert endpoints ----

def generate(student_id, test_id):
    key = f"{student_id}_test{test_id}"
    while key in active_tests:
        success, frame = camera.read()
        if not success:
            break
        frame, event = detect_cheating(frame)
        if event:
            cheating_warnings_per_test[key] += 1
            cheating_messages[key] = f"⚠️ Warning {cheating_warnings_per_test[key]}: {event}"
            if cheating_warnings_per_test[key] >= 30:
                cheating_messages[key] = "❌ Test Terminated Due To Cheating"
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    sid = session.get('student_id')
    test_id = request.args.get("test_id")
    if not sid or not test_id:
        return redirect(url_for('login'))
    return Response(generate(sid, test_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def alerts():
    sid = session.get('student_id')
    test_id = request.args.get("test_id")
    key = f"{sid}_test{test_id}"
    def stream():
        prev = ""
        while key in active_tests:
            msg = cheating_messages.get(key, "")
            if msg != prev:
                yield f"data:{msg}\n\n"
                prev = msg
    return Response(stream(), mimetype='text/event-stream')

# ---- Result ----

@app.route('/result')
def result():
    sid = session.get('student_id')
    if not sid:
        return redirect(url_for('login'))
    test_id = int(request.args.get("test_id", 1))
    key = f"{sid}_test{test_id}"
    active_tests.pop(key, None)

    questions = tests.get(test_id, [])
    submitted = request.args.to_dict()
    score = sum(1 for i, q in enumerate(questions,1)
                if submitted.get(f"q{i}") == q['answer'])
    warnings = cheating_warnings_per_test.get(key, 0)
    if warnings >= 30:
        status = "FAIL ❌ (Cheating)"
        score = 0
    else:
        status = "PASS ✅" if score >= 15 else "FAIL ❌"

    data = {"student_id": sid, "test_id": test_id,
            "score": score, "status": status, "warnings": warnings}

    all_results = json.load(open("results.json")) if os.path.exists("results.json") else {}
    if key not in all_results:
        all_results[key] = data
        json.dump(all_results, open("results.json", "w"), indent=2)

    return render_template("result.html",
                           result=status,
                           warnings=warnings,
                           score=score)

# ---- Admin dashboard ----

@app.route('/admin')
def admin():
    if session.get('student_id') != 'admin':
        return redirect(url_for('login'))
    results = json.load(open("results.json")) if os.path.exists("results.json") else {}
    return render_template("admin.html",
                           results=results,
                           active=active_tests)

# ---- Run ----

if __name__ == '__main__':
    app.run(debug=True)
