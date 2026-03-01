# 🚀 Beta 5.5 Setup Guide (Mac Studio / Apple Silicon)

This guide provides the steps to set up the **Beta 5.5 Transformer Prototype** on a new Mac Studio.

---

## 🛠️ 1. Prerequisites

Ensure the following are installed on your Mac Studio:

- **Homebrew**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **Python 3.11**: `brew install python@3.11`
- **Node.js (v22)**: `brew install node@22`
- **Docker Desktop**: [Download here](https://www.docker.com/products/docker-desktop/) (Required for TimescaleDB)
- **Git**: `brew install git`

---

## ⚡ Quick Start (Automated Script)

We have created an automated script to handle steps 2-3 for you.

1.  Create a new file named `setup.sh` on your Mac Studio.
2.  Paste the content of `setup_mac_studio.sh` (found in the root of the repo).
3.  Run it: `sh setup.sh`

---

## 🏗️ 2. Environment Setup (Manual)

### 2.1 Clone and Prepare Backend
```bash
cd backend
# Create environment file
cp .env.example .env

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2.2 Prepare Web UI
```bash
cd ../web-ui
# Install Node dependencies
npm install
```

---

## 💾 3. Database Initialization

### 3.1 Start TimescaleDB (Docker)
Ensure Docker Desktop is running, then:
```bash
cd backend
docker-compose up -d
```

### 3.2 Initialize App Schemas (SQLite & PostgreSQL)
This script initializes the dual-write system and creates the necessary tables.
```bash
python3 scripts/init_db.py
```

---

## 🧠 4. Model "Cold Start"

Since model files (`.h5`) are not in Git, you must generate them:
1. Start the system (see below).
2. Open the **Correction Studio** at `http://localhost:8503`.
3. Load a sample training file from `data/raw/` (e.g., `HK001_jessica_...`).
4. Click **"⚡ Apply All & Train"** to generate the local models in `backend/models/`.

---

## 🏁 5. Running the Platform

Use the main startup script to launch all services.

```bash
chmod +x start.sh
./start.sh
```

### Accessing the Interfaces:
- **Resident Dashboard**: [http://localhost:3002](http://localhost:3002)
- **AI Correction Studio**: [http://localhost:8503](http://localhost:8503)

---

## ⚠️ Troubleshooting (Mac Silicon)

- **Metal Acceleration**: The system is optimized for Apple Silicon (Metal). If you see ML performance issues, ensure `tensorflow-metal` is installed (`pip install tensorflow-metal`).
- **Port Conflicts**: If port 3002 is busy, ensure any legacy Beta 4 or Beta 5 sessions are stopped using `./stop.sh`.
- **Database Connection**: If the backend cannot connect to PostgreSQL, verify the `POSTGRES_PASSWORD` in `backend/.env` matches your `docker-compose.yml`.
