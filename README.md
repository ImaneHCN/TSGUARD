# TSGuard: Time-Series Guard for Data Imputation in Satellite Data

## 📌 Project Overview
TSGuard is an advanced AI-driven framework for imputing missing time-series data in satellite observations. It leverages Graph Neural Networks (GNNs) and statistical methods to enhance the accuracy of satellite-based environmental monitoring.

## 📂 Project Structure
```
TSGuard/
│-- app.py                   # Main Streamlit app
│-- requirements.txt         # Python dependencies
│-- README.md                # Project documentation
│-- models/                  # Machine learning models and simulations
│   │-- gnn_model.py         # GNN-based model for time-series imputation
│   │-- simulation.py        # Data simulation utilities
│   │-- sim_helper.py        # Helper functions for simulations
│-- utils/                   # Utility functions
│   │-- visualization.py     # Data visualization functions
│   │-- config.py            # Configuration settings
│-- docs/screenshots/        # Screenshots for documentation
│-- data/                    # Placeholder for dataset files
```

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/ImaneHCN/TSGuard.git
cd TSGuard
```

### **2️⃣ Create a Virtual Environment**
#### **Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```
#### **Mac/Linux:**
```sh
python -m venv venv
source venv/bin/activate
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

## ▶️ Running the Application
```sh
streamlit run app.py
```

## 📊 Screenshots
Below are some screenshots of the TSGuard interface:

![Dashboard](docs/screenshots/dashboard.png)
*Main dashboard displaying sensor data.*

![Visualization](docs/screenshots/visualization.png)
*Example visualization of satellite time-series data.*

## 🛠️ Features
✅ **AI-Powered Data Imputation** - Uses GNN to predict missing values.  
✅ **Interactive Visualization** - Displays time-series data in an intuitive way.  
✅ **Simulation Capabilities** - Allows testing with synthetic datasets.  

## 🤝 Contributing
Feel free to fork this repository and submit pull requests! 🚀

## 📄 License
This project is licensed under the MIT License.

## 📧 Contact
For any inquiries, please contact [your email/contact info].

