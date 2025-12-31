#!/bin/bash
# Deploy Voice Assistant to Raspberry Pi
# Usage: ./deploy.sh [options]
#
# Options:
#   --ui        Deploy UI files only
#   --wakeword  Deploy wakeword and related modules
#   --all       Deploy everything (default if no option)
#   --env       Deploy .env file
#   --restart   Restart the service only
#   --logs      Show live service logs after deploy
#   --dry-run   Show what would be deployed
#
# Prerequisites:
#   1. SSH key-based authentication set up with your Pi
#   2. rsync installed
#
# Examples:
#   ./deploy.sh --all              # Deploy everything and restart
#   ./deploy.sh --ui               # Deploy only UI changes
#   ./deploy.sh --wakeword --logs  # Deploy wakeword and show logs
#   ./deploy.sh --env --restart    # Deploy .env and restart

set -e

# === CONFIGURATION ===
# Edit these or create deploy.config to override
PI_HOST="${PI_HOST:-pi@raspberrypi.local}"
PI_PATH="${PI_PATH:-/home/pi/voice_assist}"
SERVICE_NAME="${SERVICE_NAME:-voice-assistant}"
UI_SERVICE_NAME="${UI_SERVICE_NAME:-voice-assistant-ui}"

# Load config file if exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/deploy.config" ]]; then
    source "$SCRIPT_DIR/deploy.config"
fi

# === COLORS ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === FLAGS ===
DEPLOY_UI=false
DEPLOY_WAKEWORD=false
DEPLOY_ALL=false
DEPLOY_ENV=false
DO_RESTART=false
DO_RESTART_UI=false
SHOW_LOGS=false
DRY_RUN=false

# === PARSE ARGS ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --ui) DEPLOY_UI=true; shift ;;
        --wakeword) DEPLOY_WAKEWORD=true; shift ;;
        --all) DEPLOY_ALL=true; shift ;;
        --env) DEPLOY_ENV=true; shift ;;
        --restart) DO_RESTART=true; shift ;;
        --logs) SHOW_LOGS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: ./deploy.sh [--ui] [--wakeword] [--all] [--env] [--restart] [--logs] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# === HELPER FUNCTIONS ===
log() {
    echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"
}

log_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

log_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}$1${NC}"
}

test_ssh() {
    log "Testing SSH connection to $PI_HOST..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$PI_HOST" "echo ok" &>/dev/null; then
        log_success "SSH connection OK"
        return 0
    else
        log_error "SSH connection failed"
        echo "  Make sure:"
        echo "    1. Pi is powered on and connected to network"
        echo "    2. SSH key authentication is configured"
        echo "    3. Hostname/IP is correct: $PI_HOST"
        return 1
    fi
}

rsync_deploy() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [[ ! -e "$SCRIPT_DIR/$src" ]]; then
        log_error "$src (not found)"
        return 1
    fi
    
    local rsync_opts="-avz --progress"
    if [[ "$DRY_RUN" == "true" ]]; then
        rsync_opts="$rsync_opts --dry-run"
    fi
    
    # Add trailing slash for directories to sync contents
    if [[ -d "$SCRIPT_DIR/$src" ]]; then
        src="${src%/}/"
    fi
    
    rsync $rsync_opts "$SCRIPT_DIR/$src" "${PI_HOST}:${PI_PATH}/${dest}"
    log_success "$name"
}

deploy_ui() {
    log "Deploying UI files..."
    rsync_deploy "ui/" "ui" "ui/"
    rsync_deploy "ui_server.py" "ui_server.py" "ui_server.py"
}

deploy_wakeword() {
    log "Deploying Wakeword and modules..."
    rsync_deploy "wakeword.py" "wakeword.py" "wakeword.py"
    rsync_deploy "core/" "core" "core/"
    rsync_deploy "tools/" "tools" "tools/"
    rsync_deploy "adapters/" "adapters" "adapters/"
    rsync_deploy "schemas/" "schemas" "schemas/"
}

deploy_config() {
    log "Deploying config files..."
    rsync_deploy "requirements_pi.txt" "requirements_pi.txt" "requirements_pi.txt"
    rsync_deploy "voice-assistant.service" "voice-assistant.service" "voice-assistant.service"
    rsync_deploy "voice-assistant-ui.service" "voice-assistant-ui.service" "voice-assistant-ui.service"
}

deploy_env() {
    log "Deploying environment file..."
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        log_warn "  .env not found (skipping)"
        return 0
    fi
    rsync_deploy ".env" ".env" ".env"
}

restart_service() {
    log "Restarting $SERVICE_NAME service..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "[DRY-RUN] Would restart service"
        return
    fi
    
    ssh "$PI_HOST" "sudo systemctl restart $SERVICE_NAME"
    sleep 2
    
    local status=$(ssh "$PI_HOST" "sudo systemctl is-active $SERVICE_NAME")
    if [[ "$status" == "active" ]]; then
        log_success "Service restarted successfully"
    else
        log_error "Service may have failed. Run with --logs to see output"
    fi
}

restart_ui_service() {
    log "Restarting $UI_SERVICE_NAME service..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "[DRY-RUN] Would restart UI service"
        return
    fi

    ssh "$PI_HOST" "sudo systemctl restart $UI_SERVICE_NAME"
}

show_logs() {
    log "Showing live logs (Ctrl+C to exit)..."
    ssh "$PI_HOST" "sudo journalctl -u $SERVICE_NAME -f --no-hostname -n 50"
}

# === MAIN ===
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Voice Assistant Pi Deployment        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# Default to --all if nothing specified
if ! $DEPLOY_UI && ! $DEPLOY_WAKEWORD && ! $DEPLOY_ALL && ! $DEPLOY_ENV && ! $DO_RESTART && ! $SHOW_LOGS; then
    echo "No options specified. Use --help for usage."
    echo ""
    echo "Quick start:"
    echo "  ./deploy.sh --all       Deploy everything and restart"
    echo "  ./deploy.sh --ui        Deploy UI only (no restart needed)"
    echo "  ./deploy.sh --wakeword  Deploy wakeword code and restart"
    echo ""
    exit 0
fi

# Test SSH
test_ssh || exit 1

# Deploy based on flags
if $DEPLOY_ALL; then
    deploy_ui
    deploy_wakeword
    deploy_config
    DEPLOY_ENV=true
    DO_RESTART=true
    DO_RESTART_UI=true
elif $DEPLOY_UI; then
    deploy_ui
    DO_RESTART_UI=true
elif $DEPLOY_WAKEWORD; then
    deploy_wakeword
    DEPLOY_ENV=true
    DO_RESTART=true
fi

if $DEPLOY_ENV; then
    deploy_env
fi

# Restart if needed
if $DO_RESTART && ! $SHOW_LOGS; then
    restart_service
fi

# Restart UI service if needed
if $DO_RESTART_UI && ! $SHOW_LOGS; then
    restart_ui_service
fi

# Show logs if requested
if $SHOW_LOGS; then
    if $DO_RESTART; then
        restart_service
    fi
    if $DO_RESTART_UI; then
        restart_ui_service
    fi
    show_logs
fi

echo ""
log_success "Deployment complete!"
