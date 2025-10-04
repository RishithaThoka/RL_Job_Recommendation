from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
from datetime import datetime
import uuid

# Import the RL system components (assumes the previous code is in rl_job_system.py)
# If you saved the previous code, import from that file
# For this example, we'll assume all classes are available

# Import necessary components
from pathlib import Path
import sys

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for training status and system instances
training_status = {}
system_instances = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume upload"""
    try:
        if 'resume' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['resume']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Use PDF, DOCX, or TXT'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Process resume
        from rl_job_system import ResumeUploader, ResumeProcessor
        
        uploader = ResumeUploader()
        processor = ResumeProcessor()
        
        resume_text = uploader.upload_resume(filepath)
        user_profile = processor.extract_features(resume_text)
        
        # Store profile
        profile_data = {
            'session_id': session_id,
            'filename': filename,
            'skills': user_profile['skills'],
            'experience': user_profile['experience_score'],
            'education': user_profile['education_score'],
            'keywords': user_profile['job_keywords'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save profile
        with open(os.path.join(RESULTS_FOLDER, f'{session_id}_profile.json'), 'w') as f:
            json.dump(profile_data, f)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'profile': profile_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search-jobs', methods=['POST'])
def search_jobs():
    """Search for jobs"""
    try:
        data = request.json
        session_id = data.get('session_id')
        location = data.get('location', 'remote')
        max_jobs = data.get('max_jobs', 50)
        api_keys = data.get('api_keys', {})
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
        
        # Load user profile
        profile_path = os.path.join(RESULTS_FOLDER, f'{session_id}_profile.json')
        if not os.path.exists(profile_path):
            return jsonify({'success': False, 'error': 'User profile not found'}), 404
        
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Initialize job searcher
        from rl_job_system import RealJobSearcher, API_CONFIG
        
        # Use provided API keys or defaults
        config = API_CONFIG.copy()
        if api_keys:
            config.update(api_keys)
        
        searcher = RealJobSearcher(config)
        
        # Search jobs
        jobs = searcher.search_jobs(
            keywords=profile_data['keywords'],
            location=location,
            max_results=max_jobs
        )
        
        # Save jobs
        jobs_data = {
            'session_id': session_id,
            'location': location,
            'total_jobs': len(jobs),
            'jobs': jobs,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_FOLDER, f'{session_id}_jobs.json'), 'w') as f:
            json.dump(jobs_data, f)
        
        return jsonify({
            'success': True,
            'total_jobs': len(jobs),
            'jobs': jobs[:20],  # Return first 20 for preview
            'sources': count_sources(jobs)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Start training the RL model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        episodes = data.get('episodes', 100)
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
        
        # Load user profile and jobs
        profile_path = os.path.join(RESULTS_FOLDER, f'{session_id}_profile.json')
        jobs_path = os.path.join(RESULTS_FOLDER, f'{session_id}_jobs.json')
        
        if not os.path.exists(profile_path) or not os.path.exists(jobs_path):
            return jsonify({'success': False, 'error': 'Required data not found'}), 404
        
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        with open(jobs_path, 'r') as f:
            jobs_data = json.load(f)
        
        # Initialize training status
        training_status[session_id] = {
            'status': 'starting',
            'progress': 0,
            'current_episode': 0,
            'total_episodes': episodes,
            'average_reward': 0,
            'average_feedback': 0,
            'epsilon': 1.0,
            'started_at': datetime.now().isoformat()
        }
        
        # Start training in background thread
        thread = threading.Thread(
            target=train_model_background,
            args=(session_id, profile_data, jobs_data['jobs'], episodes)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started',
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def train_model_background(session_id, profile_data, jobs, episodes):
    """Background training function"""
    try:
        from rl_job_system import (
            RLJobRecommendationSystem, 
            JobRecommendationEnvironment,
            AdvancedDQNAgent,
            API_CONFIG
        )
        import numpy as np
        
        # Update status
        training_status[session_id]['status'] = 'training'
        
        # Reconstruct user profile
        user_profile = {
            'skills': profile_data['skills'],
            'experience_score': profile_data['experience'],
            'education_score': profile_data['education'],
            'job_keywords': profile_data['keywords']
        }
        
        # Initialize environment
        env = JobRecommendationEnvironment(jobs, user_profile)
        agent = AdvancedDQNAgent(
            state_size=env.state_size,
            action_size=len(jobs)
        )
        
        # Training loop
        rewards_history = []
        feedback_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_feedback = []
            steps = 0
            
            while steps < 10:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory.buffer) > agent.batch_size:
                    agent.replay()
                
                episode_reward += reward
                if 'feedback' in info:
                    episode_feedback.append(info['feedback'])
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            rewards_history.append(episode_reward)
            if episode_feedback:
                feedback_history.append(np.mean(episode_feedback))
            
            # Update status
            progress = int((episode + 1) / episodes * 100)
            training_status[session_id].update({
                'status': 'training',
                'progress': progress,
                'current_episode': episode + 1,
                'average_reward': float(np.mean(rewards_history[-10:])),
                'average_feedback': float(np.mean(feedback_history[-10:])) if feedback_history else 0,
                'epsilon': float(agent.epsilon)
            })
            
            time.sleep(0.1)  # Small delay to prevent CPU overload
        
        # Save model
        model_path = os.path.join(MODELS_FOLDER, f'{session_id}_model.pth')
        agent.save_model(model_path)
        
        # Save training history
        history_data = {
            'session_id': session_id,
            'rewards': rewards_history,
            'feedback': feedback_history,
            'episodes': episodes,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_FOLDER, f'{session_id}_history.json'), 'w') as f:
            json.dump(history_data, f)
        
        # Store system instance
        system_instances[session_id] = {
            'agent': agent,
            'env': env
        }
        
        # Update status
        training_status[session_id].update({
            'status': 'completed',
            'progress': 100,
            'completed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        training_status[session_id].update({
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/training-status/<session_id>', methods=['GET'])
def get_training_status(session_id):
    """Get training status"""
    if session_id not in training_status:
        return jsonify({'success': False, 'error': 'Session not found'}), 404
    
    return jsonify({
        'success': True,
        'status': training_status[session_id]
    })

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get job recommendations"""
    try:
        data = request.json
        session_id = data.get('session_id')
        num_recommendations = data.get('num_recommendations', 10)
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
        
        # Check if training is complete
        if session_id not in training_status or training_status[session_id]['status'] != 'completed':
            return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
        
        # Get recommendations
        if session_id in system_instances:
            env = system_instances[session_id]['env']
            agent = system_instances[session_id]['agent']
            
            # Get recommendations
            state = env.reset()
            recommendations = []
            original_epsilon = agent.epsilon
            agent.epsilon = 0  # No exploration
            
            for _ in range(num_recommendations):
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                if 'job' in info:
                    job = info['job'].copy()
                    job['reward_score'] = float(reward)
                    job['predicted_feedback'] = info.get('feedback', 0)
                    recommendations.append(job)
                
                state = next_state
                if done:
                    break
            
            agent.epsilon = original_epsilon
            
            # Save recommendations
            recs_data = {
                'session_id': session_id,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(RESULTS_FOLDER, f'{session_id}_recommendations.json'), 'w') as f:
                json.dump(recs_data, f)
            
            return jsonify({
                'success': True,
                'recommendations': recommendations
            })
        else:
            return jsonify({'success': False, 'error': 'System instance not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.json
        session_id = data.get('session_id')
        feedback_data = data.get('feedback', [])
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
        
        # Save feedback
        feedback_file = {
            'session_id': session_id,
            'feedback': feedback_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(RESULTS_FOLDER, f'{session_id}_feedback.json'), 'w') as f:
            json.dump(feedback_file, f)
        
        # Calculate statistics
        ratings = [f['rating'] for f in feedback_data]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return jsonify({
            'success': True,
            'message': 'Feedback saved successfully',
            'statistics': {
                'average_rating': avg_rating,
                'total_feedback': len(feedback_data)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-recommendations/<session_id>', methods=['GET'])
def download_recommendations(session_id):
    """Download recommendations as JSON"""
    try:
        rec_path = os.path.join(RESULTS_FOLDER, f'{session_id}_recommendations.json')
        if not os.path.exists(rec_path):
            return jsonify({'success': False, 'error': 'Recommendations not found'}), 404
        
        return send_file(rec_path, as_attachment=True, download_name='job_recommendations.json')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training-chart/<session_id>', methods=['GET'])
def get_training_chart(session_id):
    """Get training chart data"""
    try:
        history_path = os.path.join(RESULTS_FOLDER, f'{session_id}_history.json')
        if not os.path.exists(history_path):
            return jsonify({'success': False, 'error': 'Training history not found'}), 404
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return jsonify({
            'success': True,
            'data': {
                'rewards': history['rewards'],
                'feedback': history['feedback'],
                'episodes': list(range(1, len(history['rewards']) + 1))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def count_sources(jobs):
    """Count jobs by source"""
    sources = {}
    for job in jobs:
        source = job.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    return sources

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         🚀 RL Job Recommendation System - Web Interface       ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Server starting on http://localhost:5000
    
    📋 Endpoints:
    - GET  /                           - Main web interface
    - POST /api/upload-resume          - Upload resume file
    - POST /api/search-jobs            - Search for jobs
    - POST /api/train-model            - Train RL model
    - GET  /api/training-status/<id>   - Get training progress
    - POST /api/get-recommendations    - Get recommendations
    - POST /api/submit-feedback        - Submit feedback
    - GET  /api/download-recommendations/<id> - Download results
    - GET  /api/training-chart/<id>    - Get training charts
    
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)