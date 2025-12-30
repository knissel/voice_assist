#!/bin/bash
# Deploy Voice Assistant to Raspberry Pi
# Usage: ./deploy.sh [options]
#
# Options:
#   --ui        Deploy UI files only
#   --wakeword  Deploy wakeword and related modules
#   --all       Deploy everything (default if no option)
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

set -e

# === CONFIGURATION ===
# Edit these or create deploy.config to override
PI_HOST="${PI_HOST:-pi@raspberrypi.local}"
PI_PATH="${PI_PATH:-/home/pi/voice_assist}"
SERVICE_NAME="${SERVICE_NAME:-voice-assistant}"

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
DO_RESTART=false
SHOW_LOGS=false
DRY_RUN=false

# === PARSE ARGS ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --ui) DEPLOY_UI=true; shift ;;
        --wakeword) DEPLOY_WAKEWORD=true; shift ;;
        --all) DEPLOY_ALL=true; shift ;;
        --restart) DO_RESTART=true; shift ;;
        --logs) SHOW_LOGS=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: ./deploy.sh [--ui] [--wakeword] [--all] [--restart] [--logs] [--dry-run]"
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
if ! $DEPLOY_UI && ! $DEPLOY_WAKEWORD && ! $DEPLOY_ALL && ! $DO_RESTART && ! $SHOW_LOGS; then
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
    DO_RESTART=true
elif $DEPLOY_UI; then
    deploy_ui
elif $DEPLOY_WAKEWORD; then
    deploy_wakeword
    DO_RESTART=true
fi

# Restart if needed
if $DO_RESTART && ! $SHOW_LOGS; then
    restart_service
fi

# Show logs if requested
if $SHOW_LOGS; then
    if $DO_RESTART; then
        restart_service
    fi
    show_logs
fi

echo ""
log_success "Deployment complete!"
