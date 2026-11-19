import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from utils.predict import predict_image

app = Flask(__name__)
app.secret_key = 'super_secret_healthcare_ai_key_2026'

# Global prediction history
prediction_history = []

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Simple mock authentication for demonstration
        if email == 'admin@hospital.com' and password == 'admin123':
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Use admin@hospital.com / admin123', 'danger')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
        
    if 'mri_image' not in request.files:
        flash('No image part', 'danger')
        return redirect(url_for('dashboard'))
        
    file = request.files['mri_image']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result, error = predict_image(file_path)
        
        if error:
            flash(f"Error during prediction: {error}", 'danger')
            return redirect(url_for('dashboard'))
        
        # Add to history
        from datetime import datetime
        prediction_history.append({
            'filename': filename,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().strftime('%d %b %Y %H:%M')
        })
        
        # Generate probability graph
        if 'probabilities' in result and result['probabilities']:
            fig, ax = plt.subplots(figsize=(8, 4))
            classes = [p['class'] for p in result['probabilities']]
            probs = [p['prob'] for p in result['probabilities']]
            ax.bar(classes, probs, color='skyblue')
            ax.set_title('Prediction Probabilities')
            ax.set_ylabel('Probability (%)')
            ax.set_xlabel('Classes')
            plt.xticks(rotation=45)
            graph_filename = f"graph_{os.path.splitext(filename)[0]}.png"
            graph_path = os.path.join(app.config['UPLOAD_FOLDER'], graph_filename)
            plt.tight_layout()
            plt.savefig(graph_path, format='png')
            plt.close()
            result['graph_url'] = f"uploads/{graph_filename}"
        else:
            result['graph_url'] = None
            
        return render_template('result.html', result=result, image_url=f"uploads/{filename}")
        
    flash('Invalid file type. Please upload PNG, JPG or JPEG.', 'warning')
    return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/history')
def history():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('history.html', history=prediction_history)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
