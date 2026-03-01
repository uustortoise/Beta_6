# Cloud Mobile Server Deployment Plan (Alibaba Cloud)

## 1. Executive Summary
This document outlines the strategy for deploying the **Health Advisory Chatbot** (Beta 5.5 Module) to a cloud environment (targeting Alibaba Cloud) to support the mobile application. The goal is to establish a reliable, scalable, and secure backend infrastructure.

## 2. Technical Strategy: Containerization
We will use **Docker** to package the application, ensuring it runs identically on the cloud server as it does in development.

### 2.1 Backend (Python)
*   **Base Image**: `python:3.9-slim`
*   **Server**: FastAPI (via Uvicorn) for high-performance async handling.
*   **Dependencies**: Explicitly defined in `requirements.txt`.
*   **Port**: Exposed internally (e.g., 8000).

### 2.2 Frontend (Next.js)
*   **Base Image**: `node:18-alpine`
*   **Build**: Multi-stage build to optimize image size (Builder -> Runner).
*   **Server**: Next.js production server.
*   **Port**: Exposed internally (e.g., 3000).

### 2.3 Orchestration
*   **Tool**: `docker-compose`
*   **Function**: Manages the multi-container application (Frontend + Backend + Nginx + Database).
*   **Benefit**: Single command (`docker-compose up -d`) to start the entire system.

## 3. Infrastructure Architecture

### 3.1 Server Specification (Recommended)
*   **Provider**: Alibaba Cloud (Aliyun)
*   **Service**: ECS (Elastic Compute Service) or Simple Application Server
*   **OS**: Ubuntu 22.04 LTS (Stable, widely supported)
*   **Minimum Specs**:
    *   CPU: 2 vCPU
    *   RAM: 4 GB (AI models and Next.js builds can be memory intensive)
    *   Disk: 40 GB+ ESSD
*   **Network**: Public IP required. Security Group rules to allow HTTP (80), HTTPS (443), and SSH (22).

### 3.2 Database Strategy: Evolution from Docker to Aliyun RDS

Choosing between self-managed Docker and fully-managed RDS involves balancing cost versus operational reliability.

#### Option A: Database inside Docker (Starting Phase)
*   **How it works**: We add a service (e.g., `postgres` or `mariadb`) to our `docker-compose.yml`. We use a "Docker Volume" to map the database data to a folder on the ECS disk (e.g., `/data/db`).
*   **Why start here?**:
    *   **Zero Cost**: Included in your ECS monthly bill; no extra database fees.
    *   **Total Control**: We can tweak database settings and plugins instantly.
    *   **Simplicity**: The entire app (UI + API + DB) is captured in one single file.
*   **Caveats**: You must manually manage backups (e.g., a script that uploads a DB dump to Alibaba OSS once a day).

#### Option B: Alibaba ApsaraDB RDS (Growth/Production Phase)
*   **How it works**: You provision a dedicated database instance through the Alibaba Cloud console.
*   **Why upgrade?**:
    *   **High Availability**: Automated failover; if one node dies, another takes over instantly.
    *   **Automated Backups**: Point-in-time recovery (e.g., "restore the database to exactly 10:45 AM yesterday").
    *   **Security**: Dedicated VPC (isolated network) and built-in DDoS protection.
    *   **Performance**: Dedicated RAM/CPU that isn't shared with the web server.

#### The Migration Path
Migrating is straightforward when we are ready:
1.  **Export**: Run a "dump" command on the Docker database.
2.  **Import**: Upload the dump file to the RDS instance.
3.  **Config**: Update the `DATABASE_URL` in the `.env` file to point to the new RDS endpoint.
4.  **Shutdown**: Stop the Docker database container.

### 3.3 Connectivity & Security
*   **Reverse Proxy (Nginx)**: An Nginx container will sit in front of the services.
    *   Directs `/api/*` traffic -> Backend Container
    *   Directs web traffic -> Frontend Container
    *   Handles SSL/TLS termination (HTTPS certificates).
*   **Database**:
    *   *Phase 1 (MVP)*: SQLite or PostgreSQL running in a Docker container (easiest setup).
    *   *Phase 2 (Scale)*: Migrate to Alibaba Cloud ApsaraDB RDS for managed backups and high availability.

## 4. Automation & Maintenance
To simplify management for the team, we will create utility scripts:

1.  `setup_server.sh`:
    *   Installs Docker & Docker Compose on a fresh Ubuntu server.
    *   Clones the repository.
    *   Sets up environment variables (`.env`).
2.  `deploy.sh`:
    *   Pulls the latest code from Git.
    *   Rebuilds containers.
    *   Restarts services with zero downtime (rolling update if possible, otherwise brief restart).

## 5. Implementation Roadmap
1.  **[ ] Dockerize List**: Create `Dockerfile` for Backend and Frontend.
2.  **[ ] Compose Setup**: Create `docker-compose.yml` linking services.
3.  **[ ] Gateway Config**: Configure `nginx.conf` for routing.
4.  **[ ] Scripting**: Write the `setup_server.sh` automation script.
5.  **[ ] Testing**: Verify local deployment using Docker Desktop.
6.  **[ ] Cloud Handover**: Team provisions Alibaba Cloud instance -> Run setup script.

## 6. Discussion Points
*   **Domain Name**: Do we have a domain (e.g., `chat.elderly-care.com`)? We need this for HTTPS.
*   **SSL Certificates**: We can use Let's Encrypt (free) or purchase one via Alibaba.
*   **Data Persistence**: Verify regulations regarding storing elderly care data on cloud servers (location of data center).
