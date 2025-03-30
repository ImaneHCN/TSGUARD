# Project Setup Guide

## 🚀 Setting Up the Virtual Environment
To ensure a clean and isolated development environment, follow these steps to create and activate a virtual environment before installing dependencies.

### 1️⃣ Clone the Repository
```sh
git clone <repository-url>
cd <project-directory>
```

### 2️⃣ Create a Virtual Environment
#### 🖥️ Windows
```sh
python -m venv venv
source venv\Scripts\activate
```
Or
```sh
python3 -m venv venv
source venv\Scripts\activate
```

#### 🐧 macOS / Linux
```sh
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
Once the virtual environment is activated, install all required dependencies:
```sh
pip install -r requirements.txt
```

### 4️⃣ Verify Installation
```sh
pip list
```

### 5️⃣ Deactivate the Virtual Environment (If Needed)
```sh
deactivate
```

## 🎯 Running the Project
Once the environment is set up, follow the project's specific instructions to run the application.

---

✅ **Always ensure you're working inside the virtual environment to prevent dependency conflicts!**

