# rl_job_system.py
# Complete RL Job Recommendation System Module

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
import re
import os
import PyPDF2
import docx
import requests
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# API Configuration
API_CONFIG = {
    'rapidapi_key': 'YOUR_RAPIDAPI_KEY_HERE',
    'reed_api_key': 'YOUR_REED_API_KEY_HERE',
    'adzuna_app_id': 'YOUR_ADZUNA_APP_ID_HERE',
    'adzuna_app_key': 'YOUR_ADZUNA_APP_KEY_HERE'
}

class ResumeUploader:
    """Handles resume file upload and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def upload_resume(self, file_path: str) -> str:
        """Upload and extract text from resume file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported: {self.supported_formats}")
        
        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension == '.docx':
                return self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                return self._extract_txt_text(file_path)
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return text.strip()
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
        return text.strip()
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
        return text.strip()

class ResumeProcessor:
    """Process resume and extract features"""
    
    def __init__(self):
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'r', 'go', 'php', 'ruby'],
            'data_science': ['machine learning', 'deep learning', 'statistics', 'data analysis', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite'],
            'management': ['project management', 'agile', 'scrum', 'leadership', 'team lead']
        }
    
    def extract_features(self, resume_text: str) -> Dict[str, Any]:
        """Extract comprehensive features from resume"""
        resume_lower = resume_text.lower()
        
        # Extract skills
        skills = {}
        skills_detailed = {}
        for category, skill_list in self.skill_categories.items():
            found_skills = [skill for skill in skill_list if skill in resume_lower]
            skills[category] = len(found_skills)
            skills_detailed[f"{category}_skills"] = found_skills
        
        # Extract experience and education
        experience_score = self._extract_experience(resume_text)
        education_score = self._extract_education(resume_text)
        
        # Extract job keywords
        job_keywords = self._extract_job_keywords(resume_text, skills_detailed)
        
        return {
            'skills': skills,
            'skills_detailed': skills_detailed,
            'experience_score': experience_score,
            'education_score': education_score,
            'job_keywords': job_keywords,
            'resume_text': resume_text
        }
    
    def _extract_experience(self, resume_text: str) -> int:
        """Extract years of experience from resume"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\s*-\s*\d+\s*years?',
            r'experience.*?(\d+)\s*years?'
        ]
        
        experience_years = []
        resume_lower = resume_text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, resume_lower)
            for match in matches:
                try:
                    years = int(match)
                    if 0 <= years <= 50:
                        experience_years.append(years)
                except ValueError:
                    continue
        
        if experience_years:
            return max(experience_years)
        
        # Fallback: estimate based on seniority keywords
        seniority_keywords = ['senior', 'lead', 'manager', 'director', 'principal']
        keyword_count = sum(1 for kw in seniority_keywords if kw in resume_lower)
        return min(keyword_count * 2, 10)
    
    def _extract_education(self, resume_text: str) -> int:
        """Extract education level from resume"""
        education_levels = {
            'phd': 5, 'doctorate': 5, 'ph.d': 5,
            'masters': 4, 'master': 4, 'mba': 4, 'm.s': 4, 'm.a': 4,
            'bachelor': 3, 'bachelors': 3, 'b.s': 3, 'b.a': 3, 'b.tech': 3,
            'associate': 2, 'diploma': 2,
            'certificate': 1
        }
        
        resume_lower = resume_text.lower()
        max_education = 0
        
        for degree, level in education_levels.items():
            if degree in resume_lower:
                max_education = max(max_education, level)
        
        return max_education
    
    def _extract_job_keywords(self, resume_text: str, skills_detailed: Dict) -> List[str]:
        """Extract relevant job search keywords"""
        keywords = []
        
        # Add top skills from each category
        for category, skill_list in skills_detailed.items():
            if skill_list:
                keywords.extend(skill_list[:2])  # Top 2 skills per category
        
        # Add job title patterns
        job_patterns = [
            r'(?:software|data|web|machine learning)\s+(?:engineer|developer|scientist|analyst)',
            r'(?:senior|junior|lead)\s+(?:engineer|developer)',
            r'(?:full\s+stack|frontend|backend)\s+developer'
        ]
        
        resume_lower = resume_text.lower()
        for pattern in job_patterns:
            matches = re.findall(pattern, resume_lower)
            keywords.extend(matches)
        
        # Remove duplicates and limit
        keywords = list(set(keywords))[:8]
        
        # Default keywords if none found
        if not keywords:
            keywords = ['software engineer', 'developer']
        
        return keywords

class RealJobSearcher:
    """Search for jobs using real APIs"""
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_config = api_config
        self.cache = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_jobs(self, keywords: List[str], location: str = "remote", 
                   max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for jobs using multiple sources"""
        
        cache_key = f"{'-'.join(keywords)}_{location}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, jobs = self.cache[cache_key]
            if time.time() - cache_time < 3600:  # 1 hour
                return jobs
        
        all_jobs = []
        
        # Try real APIs if configured
        try:
            if self._has_valid_key('rapidapi_key'):
                print("Searching JSearch API...")
                jsearch_jobs = self._search_jsearch(keywords, location, max_results//3)
                all_jobs.extend(jsearch_jobs)
        except Exception as e:
            print(f"JSearch API error: {e}")
        
        # Fallback to mock data
        if len(all_jobs) < 10:
            print("Using mock data...")
            all_jobs.extend(self._get_mock_jobs(keywords, location, max_results))
        
        # Remove duplicates
        unique_jobs = self._remove_duplicates(all_jobs)
        final_jobs = unique_jobs[:max_results]
        
        # Cache results
        self.cache[cache_key] = (time.time(), final_jobs)
        
        return final_jobs
    
    def _has_valid_key(self, key_name: str) -> bool:
        """Check if API key is configured"""
        key = self.api_config.get(key_name, '')
        return key and 'YOUR_' not in key and len(key) > 10
    
    def _search_jsearch(self, keywords: List[str], location: str, limit: int) -> List[Dict]:
        """Search using JSearch API"""
        jobs = []
        url = "https://jsearch.p.rapidapi.com/search"
        
        headers = {
            "X-RapidAPI-Key": self.api_config['rapidapi_key'],
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        
        for keyword in keywords[:2]:  # Limit to save quota
            params = {
                "query": f"{keyword} {location}",
                "page": "1",
                "num_pages": "1"
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for job in data.get('data', [])[:limit//2]:
                        parsed = self._parse_jsearch_job(job)
                        if parsed:
                            jobs.append(parsed)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"JSearch request failed: {e}")
                continue
        
        return jobs
    
    def _parse_jsearch_job(self, job_data: Dict) -> Dict:
        """Parse JSearch API response"""
        try:
            return {
                'id': job_data.get('job_id', f"js_{random.randint(1000, 9999)}"),
                'title': job_data.get('job_title', 'Position'),
                'company': job_data.get('employer_name', 'Company'),
                'description': (job_data.get('job_description', '') or '')[:500],
                'location': job_data.get('job_city', 'Remote'),
                'salary': self._extract_salary(job_data.get('job_min_salary'), job_data.get('job_max_salary')),
                'url': job_data.get('job_apply_link', ''),
                'posted_date': (job_data.get('job_posted_at_datetime_utc', '') or '')[:10],
                'job_type': job_data.get('job_employment_type', 'Full-time'),
                'required_skills': self._extract_skills_from_desc(job_data.get('job_description', '') or ''),
                'experience_required': self._extract_exp_from_desc(job_data.get('job_description', '') or ''),
                'category': 'software_engineering',
                'source': 'JSearch'
            }
        except Exception as e:
            print(f"Error parsing job: {e}")
            return None
    
    def _extract_salary(self, min_sal, max_sal) -> int:
        """Extract salary from min/max values"""
        try:
            if min_sal and max_sal:
                return (int(min_sal) + int(max_sal)) // 2
            elif min_sal:
                return int(min_sal)
            elif max_sal:
                return int(max_sal)
        except:
            pass
        return random.randint(60000, 120000)
    
    def _extract_skills_from_desc(self, description: str) -> Dict[str, int]:
        """Extract required skills from job description"""
        skills = {
            'programming': 0, 'data_science': 0, 'web_dev': 0,
            'cloud': 0, 'database': 0, 'management': 0
        }
        
        desc_lower = description.lower()
        
        if any(w in desc_lower for w in ['python', 'java', 'javascript', 'programming']):
            skills['programming'] = 3
        if any(w in desc_lower for w in ['machine learning', 'data science', 'ai', 'analytics']):
            skills['data_science'] = 3
        if any(w in desc_lower for w in ['react', 'angular', 'web development', 'frontend', 'backend']):
            skills['web_dev'] = 3
        if any(w in desc_lower for w in ['aws', 'azure', 'cloud', 'docker', 'kubernetes']):
            skills['cloud'] = 2
        if any(w in desc_lower for w in ['sql', 'database', 'mongodb', 'postgresql']):
            skills['database'] = 2
        if any(w in desc_lower for w in ['manager', 'lead', 'management', 'leadership']):
            skills['management'] = 2
        
        return skills
    
    def _extract_exp_from_desc(self, description: str) -> int:
        """Extract experience requirement from description"""
        patterns = [r'(\d+)\+?\s*years?', r'(\d+)-\d+\s*years?']
        
        for pattern in patterns:
            matches = re.findall(pattern, description.lower())
            if matches:
                try:
                    return int(matches[0])
                except:
                    pass
        
        return random.randint(2, 5)
    
    def _get_mock_jobs(self, keywords: List[str], location: str, limit: int) -> List[Dict]:
        """Generate mock jobs for testing"""
        companies = ['TechCorp', 'DataInc', 'WebSolutions', 'CloudTech', 'AILabs', 'StartupXYZ', 'DevCo', 'InnovateTech']
        job_types = ['Full-time', 'Part-time', 'Contract', 'Remote']
        
        jobs = []
        
        for i, keyword in enumerate(keywords[:5]):
            for j in range(min(limit // len(keywords), 10)):
                job_id = f"mock_{i}_{j}_{random.randint(1000, 9999)}"
                
                jobs.append({
                    'id': job_id,
                    'title': f"{random.choice(['Senior', 'Junior', 'Lead'])} {keyword.title()} {random.choice(['Engineer', 'Developer', 'Specialist'])}",
                    'company': random.choice(companies),
                    'description': f"Exciting opportunity for {keyword} professional. Work on cutting-edge projects using modern technologies. Join our dynamic team!",
                    'location': location,
                    'salary': random.randint(60000, 150000),
                    'url': f'https://example.com/job/{job_id}',
                    'posted_date': '2024-12-15',
                    'job_type': random.choice(job_types),
                    'required_skills': self._extract_skills_from_desc(keyword),
                    'experience_required': random.randint(2, 6),
                    'category': 'software_engineering',
                    'source': 'Mock'
                })
        
        return jobs
    
    def _remove_duplicates(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate job listings"""
        seen = set()
        unique = []
        
        for job in jobs:
            key = f"{job['title']}_{job['company']}"
            if key not in seen:
                seen.add(key)
                unique.append(job)
        
        return unique

class DQNNetwork(nn.Module):
    """Deep Q-Network for RL"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) < batch_size:
            return None
        
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class AdvancedDQNAgent:
    """Advanced DQN Agent with target network"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.005
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.memory = PrioritizedReplayBuffer(10000)
        self.batch_size = 64
        self.losses = []
    
    def get_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = self.get_state_tensor(state)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().data.numpy().argmax()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        batch_data = self.memory.sample(self.batch_size, beta=0.4)
        if batch_data is None:
            return
        
        experiences, indices, weights = batch_data
        
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).detach().max(1)[0]
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        td_errors = target_q.unsqueeze(1) - current_q
        loss = (weights_tensor * td_errors.pow(2).squeeze()).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.memory.update_priorities(indices, td_errors.cpu().data.numpy().flatten())
        self.soft_update()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
    
    def soft_update(self):
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class JobRecommendationEnvironment:
    """RL Environment for job recommendations"""
    
    def __init__(self, jobs: List[Dict[str, Any]], user_profile: Dict[str, Any]):
        self.all_jobs = jobs
        self.user_profile = user_profile
        self.recommended_jobs = set()
        self.feedback_history = []
        self.current_episode_jobs = []
        self.max_recommendations = 10
        self.state_size = 25
    
    def reset(self) -> np.ndarray:
        self.recommended_jobs = set()
        self.feedback_history = []
        self.current_episode_jobs = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        user_skills = list(self.user_profile['skills'].values())
        user_exp = [self.user_profile['experience_score'] / 10.0]
        user_edu = [self.user_profile['education_score'] / 5.0]
        
        recent_feedback = self.feedback_history[-5:] if self.feedback_history else [0] * 5
        recent_feedback = [(f / 5.0) for f in recent_feedback]
        recent_feedback += [0] * (5 - len(recent_feedback))
        
        num_recommended = [len(self.recommended_jobs) / self.max_recommendations]
        
        categories = [job['category'] for job in self.current_episode_jobs]
        diversity = [len(set(categories)) / max(len(categories), 1)]
        avg_score = [np.mean([job.get('recommendation_score', 0.5) for job in self.current_episode_jobs])] if self.current_episode_jobs else [0.5]
        
        state = user_skills + user_exp + user_edu + recent_feedback + num_recommended + diversity + avg_score
        state = state[:self.state_size]
        state += [0] * (self.state_size - len(state))
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action >= len(self.all_jobs):
            return self._get_state(), -1.0, False, {'error': 'Invalid action'}
        
        job = self.all_jobs[action]
        
        if job['id'] in self.recommended_jobs:
            reward = -0.5
            feedback = 0
        else:
            reward = self._calculate_reward(job)
            feedback = self._simulate_feedback(reward)
            self.recommended_jobs.add(job['id'])
            self.current_episode_jobs.append(job)
            self.feedback_history.append(feedback)
        
        done = len(self.recommended_jobs) >= self.max_recommendations
        
        info = {
            'job': job,
            'feedback': feedback,
            'total_recommended': len(self.recommended_jobs)
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, job: Dict[str, Any]) -> float:
        reward = 0.0
        
        # Skill matching
        user_skills = self.user_profile['skills']
        job_skills = job.get('required_skills', {})
        
        for skill_cat, required in job_skills.items():
            user_level = user_skills.get(skill_cat, 0)
            if user_level >= required:
                reward += 0.5
            elif user_level > 0:
                reward += 0.2
            else:
                reward -= 0.2
        
        # Experience matching
        job_exp = job.get('experience_required', 0)
        user_exp = self.user_profile['experience_score']
        exp_diff = abs(user_exp - job_exp)
        
        if exp_diff <= 1:
            reward += 0.8
        elif exp_diff <= 2:
            reward += 0.4
        else:
            reward -= 0.2
        
        # Diversity bonus
        if len(self.current_episode_jobs) > 0:
            categories = [j['category'] for j in self.current_episode_jobs]
            if job['category'] not in categories:
                reward += 0.2
        
        return reward
    
    def _simulate_feedback(self, reward: float) -> int:
        base = np.clip((reward + 2) * 1.25, 0, 5)
        noise = np.random.normal(0, 0.5)
        return int(np.clip(base + noise, 0, 5))

class RLJobRecommendationSystem:
    """Complete RL-based Job Recommendation System"""
    
    def __init__(self, api_config: Dict[str, str]):
        self.resume_uploader = ResumeUploader()
        self.job_searcher = RealJobSearcher(api_config)
        self.resume_processor = ResumeProcessor()
        self.user_profile = None
        self.available_jobs = []
        self.agent = None
        self.env = None
        self.training_history = {
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'average_feedback': []
        }
    
    def upload_and_process_resume(self, file_path: str) -> Dict[str, Any]:
        """Upload and process resume"""
        print(f"Uploading resume from: {file_path}")
        
        resume_text = self.resume_uploader.upload_resume(file_path)
        print(f"Extracted {len(resume_text)} characters from resume")
        
        self.user_profile = self.resume_processor.extract_features(resume_text)
        
        print("Resume processed successfully!")
        print(f"Skills: {self.user_profile['skills']}")
        print(f"Experience: {self.user_profile['experience_score']} years")
        print(f"Education: {self.user_profile['education_score']}/5")
        
        return self.user_profile
    
    def search_jobs(self, location: str = "remote", max_jobs: int = 50):
        """Search for jobs"""
        print(f"Searching for jobs...")
        print(f"Keywords: {self.user_profile['job_keywords']}")
        print(f"Location: {location}")
        
        self.available_jobs = self.job_searcher.search_jobs(
            keywords=self.user_profile['job_keywords'],
            location=location,
            max_results=max_jobs
        )
        
        print(f"Found {len(self.available_jobs)} jobs")
        
        sources = {}
        for job in self.available_jobs:
            source = job.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"Sources: {sources}")
        
        return self.available_jobs
    
    def train_rl_agent(self, episodes: int = 100, save_model: bool = True):
        """Train the RL agent"""
        print(f"Training RL Agent for {episodes} episodes...")
        
        if not self.available_jobs:
            raise ValueError("No jobs available. Please search for jobs first.")
        
        # Initialize environment and agent
        self.env = JobRecommendationEnvironment(self.available_jobs, self.user_profile)
        self.agent = AdvancedDQNAgent(
            state_size=self.env.state_size,
            action_size=len(self.available_jobs)
        )
        
        best_reward = -float('inf')
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_feedback = []
            steps = 0
            
            while steps < 10:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                
                if len(self.agent.memory.buffer) > self.agent.batch_size:
                    self.agent.replay()
                
                episode_reward += reward
                if 'feedback' in info:
                    episode_feedback.append(info['feedback'])
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Track metrics
            self.training_history['rewards'].append(episode_reward)
            self.training_history['epsilon'].append(self.agent.epsilon)
            avg_feedback = np.mean(episode_feedback) if episode_feedback else 0
            self.training_history['average_feedback'].append(avg_feedback)
            
            if self.agent.losses:
                self.training_history['losses'].append(np.mean(self.agent.losses[-10:]))
            
            # Save best model
            if episode_reward > best_reward and save_model:
                best_reward = episode_reward
        
        print(f"Training completed!")
        print(f"Best Episode Reward: {best_reward:.2f}")
        print(f"Final Epsilon: {self.agent.epsilon:.3f}")
    
    def get_recommendations(self, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get job recommendations using trained agent"""
        print(f"Generating {num_recommendations} recommendations...")
        
        if self.agent is None:
            raise ValueError("Agent not trained. Please train the agent first.")
        
        # Reset environment
        state = self.env.reset()
        recommendations = []
        
        # Use trained agent without exploration
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        for _ in range(num_recommendations):
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            if 'job' in info:
                job = info['job'].copy()
                job['reward_score'] = float(reward)
                job['predicted_feedback'] = info.get('feedback', 0)
                recommendations.append(job)
            
            state = next_state
            
            if done:
                break
        
        self.agent.epsilon = original_epsilon
        
        print(f"Generated {len(recommendations)} recommendations")
        return recommendations