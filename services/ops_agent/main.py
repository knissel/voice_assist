import time
import logging
import os
import subprocess
import yaml
import requests
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ops_agent")

app = FastAPI(title="Voice Assist Ops Agent")

# CORS for Dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve config path relative to this script, allow override via env var
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config.yaml")
CONFIG_PATH = os.getenv("OPS_CONFIG", DEFAULT_CONFIG)

class ServiceConfig(BaseModel):
    type: str # 'powershell', 'systemd', 'process'
    start_cmd: Optional[str] = None
    stop_cmd: Optional[str] = None
    service_name: Optional[str] = None
    check_url: Optional[str] = None

class ServiceStatus(BaseModel):
    name: str
    status: str # 'running', 'stopped', 'error', 'unknown'
    details: Optional[str] = None

def load_config() -> Dict[str, dict]:
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Config file {CONFIG_PATH} not found.")
        return {"services": {}}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@app.get("/health")
def health_check():
    return {"status": "ok", "agent": "voice-assist-ops-agent"}

@app.get("/env")
def get_env():
    """Reads the .env file from the repo root."""
    env_path = os.path.join(BASE_DIR, "..", "..", ".env")
    if not os.path.exists(env_path):
        return {}
    
    env_data = {}
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    env_data[key.strip()] = val.strip()
    except Exception as e:
        logger.error(f"Failed to read .env: {e}")
        raise HTTPException(status_code=500, detail="Failed to read .env")
    
    return env_data

@app.post("/env")
def update_env(env_vars: Dict[str, str]):
    """Updates the .env file in the repo root."""
    env_path = os.path.join(BASE_DIR, "..", "..", ".env")
    try:
        # Read existing to preserve comments? 
        # For simplicity, we'll overwrite or append.
        # Let's just overwrite with the provided dict for clean management.
        with open(env_path, "w") as f:
            for key, val in env_vars.items():
                f.write(f"{key}={val}\n")
        logger.info(".env file updated via agent")
        return {"status": "success", "message": ".env updated"}
    except Exception as e:
        logger.error(f"Failed to write .env: {e}")
        raise HTTPException(status_code=500, detail="Failed to write .env")

@app.get("/services", response_model=List[ServiceStatus])
def list_services():
    config = load_config()
    services_config = config.get("services", {})
    status_list = []

    for name, svc in services_config.items():
        status = "unknown"
        details = ""
        
        svc_type = svc.get("type")
        check_url = svc.get("check_url")

        # 1. Try URL check first if available (most reliable for web servers)
        if check_url:
            try:
                # Short timeout so dashboard doesn't hang
                resp = requests.get(check_url, timeout=1)
                if resp.status_code == 200:
                    status = "running"
                else:
                    status = "error"
                    details = f"HTTP {resp.status_code}"
            except requests.exceptions.RequestException:
                status = "stopped"
        
        # 2. Fallback checks based on type if URL check failed or not present
        if status != "running":
            if svc_type == "systemd":
                # Check systemctl status
                service_name = svc.get("service_name")
                if service_name:
                    try:
                        res = subprocess.run(
                            ["systemctl", "is-active", service_name], 
                            capture_output=True, text=True
                        )
                        if res.returncode == 0:
                            status = "running"
                        else:
                            status = "stopped"
                    except FileNotFoundError:
                        status = "error"
                        details = "systemctl not found"
            
            elif svc_type == "powershell":
                # Hard to check generic powershell process without PID tracking
                # For now, if URL check failed, assume stopped.
                pass

        status_list.append(ServiceStatus(name=name, status=status, details=details))

    return status_list

@app.post("/services/{service_name}/{action}")
def manage_service(service_name: str, action: str):
    config = load_config()
    svc = config.get("services", {}).get(service_name)
    
    if not svc:
        raise HTTPException(status_code=404, detail="Service not found")
    
    if action not in ["start", "stop", "restart"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    svc_type = svc.get("type")

    try:
        if svc_type == "systemd":
            service_sys_name = svc.get("service_name")
            subprocess.run(["sudo", "systemctl", action, service_sys_name], check=True)
            return {"status": "success", "action": action}
        
        elif svc_type == "powershell":
            cmd = ""
            if action == "start":
                cmd = svc.get("start_cmd")
            elif action == "stop":
                cmd = svc.get("stop_cmd")
            elif action == "restart":
                # Naive restart
                stop_cmd = svc.get("stop_cmd")
                start_cmd = svc.get("start_cmd")
                if stop_cmd:
                    subprocess.run(["powershell", "-Command", stop_cmd], check=False)
                    time.sleep(2) # Give it a moment to die
                if start_cmd:
                    subprocess.run(["powershell", "-Command", start_cmd], check=True)
                return {"status": "success", "action": "restart"}

            if cmd:
                subprocess.run(["powershell", "-Command", cmd], check=True)
                return {"status": "success", "action": action}
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "error", "detail": "Action not implemented for this service type"}

@app.post("/deploy")
def deploy_updates():
    """
    Pulls the latest code from git and restarts all managed services.
    """
    try:
        # 1. Git Pull
        # Assuming the agent is running from within the repo
        logger.info("Executing git pull...")
        pull_res = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"Git pull result: {pull_res.stdout}")

        # 2. Restart Services
        config = load_config()
        services_config = config.get("services", {})
        restarted = []

        for name, svc in services_config.items():
            try:
                logger.info(f"Restarting service: {name}")
                svc_type = svc.get("type")
                if svc_type == "powershell":
                    stop_cmd = svc.get("stop_cmd")
                    start_cmd = svc.get("start_cmd")
                    if stop_cmd:
                        subprocess.run(["powershell", "-Command", stop_cmd], check=False)
                        time.sleep(2) # Give it a moment to die
                    if start_cmd:
                        subprocess.run(["powershell", "-Command", start_cmd], check=True)
                    restarted.append(name)
                
                elif svc_type == "systemd":
                    service_sys_name = svc.get("service_name")
                    subprocess.run(["sudo", "systemctl", "restart", service_sys_name], check=True)
                    restarted.append(name)
                    
            except Exception as e:
                logger.error(f"Failed to restart {name}: {e}")

        return {"status": "success", "git_output": pull_res.stdout, "restarted_services": restarted}

    except subprocess.CalledProcessError as e:
        logger.error(f"Deploy failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deploy failed: {e.stderr}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
