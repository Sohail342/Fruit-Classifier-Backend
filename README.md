# Image Classification Web App

A full-stack application for image classification using FastAPI, React, and TensorFlow.

## Project Structure

```
├── backend/               # FastAPI backend
│   ├── app/              # Application code
│   │   ├── api/          # API endpoints
│   │   ├── models/       # ML models
│   ├── requirements.txt  # Python dependencies
│   └── main.py           # Entry point
│   └── models           # Custom Train model
├── frontend/             # React frontend
│   ├── public/           # Static files
│   ├── src/              # Source code
│   ├── package.json      # Node dependencies
│   └── README.md         # Frontend documentation
└── README.md             # Project documentation
```

## Features

- Image upload and preview
- Real-time image classification using TensorFlow
- Responsive UI for desktop and mobile devices
- Classification history and results management

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Technologies Used

- **Backend**: FastAPI, TensorFlow, Python
- **Frontend**: React, JavaScript/TypeScript
- **ML/AI**: TensorFlow
