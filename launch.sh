#!/usr/bin/env bash
# Start Gradio. setup.sh must have finished first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SCENE=0
if [[ "${1:-}" == "--scene" ]]; then
  SCENE=1
fi

VENV_PY="${HERE}/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "No .venv/bin/python. Run: bash setup.sh" >&2
  exit 1
fi
# Conda (vast.ai often has `(main)` on PATH) must not steal `python`.
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="${HERE}/.venv/bin:${PATH}"
export PYTHONNOUSERSITE=1
hash -r 2>/dev/null || true

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HERE}/.libero}"
export PYTHONPATH="${HERE}/src"
if [[ -d "${HERE}/vendor/LIBERO" ]]; then
  PYTHONPATH="${PYTHONPATH}:${HERE}/vendor/LIBERO"
fi
if [[ -n "${PYTHONPATH_EXTRA:-}" ]]; then
  PYTHONPATH="${PYTHONPATH}:${PYTHONPATH_EXTRA}"
fi
export PYTHONPATH

if ! "$VENV_PY" "${HERE}/scripts/write_libero_home_config.py"; then
  echo "WARNING: could not write LIBERO config (vendor/LIBERO missing?)." >&2
fi

PORT="${ROBOENV_PORT:-${PORT:-7860}}"
export ROBOENV_PORT="$PORT"
export ROBOENV_HOST="${ROBOENV_HOST:-0.0.0.0}"

USER_NAME="${ROBOENV_USER:-vastai}"
PASS=""
PASS_SRC=""
if [[ -n "${ROBOENV_PASSWORD:-}" ]]; then
  PASS="$ROBOENV_PASSWORD"
  PASS_SRC="ROBOENV_PASSWORD"
elif [[ -n "${OPEN_BUTTON_TOKEN:-}" ]]; then
  PASS="$OPEN_BUTTON_TOKEN"
  PASS_SRC="OPEN_BUTTON_TOKEN"
elif [[ -n "${JUPYTER_TOKEN:-}" ]]; then
  PASS="$JUPYTER_TOKEN"
  PASS_SRC="JUPYTER_TOKEN"
elif [[ -f "$HERE/.gradio_token" ]]; then
  PASS="$(tr -d '[:space:]' < "$HERE/.gradio_token")"
  PASS_SRC=".gradio_token"
fi
if [[ -z "$PASS" ]]; then
  PASS="$("$VENV_PY" - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  PASS_SRC="generated .gradio_token"
  umask 077
  printf '%s\n' "$PASS" > "$HERE/.gradio_token"
  chmod 600 "$HERE/.gradio_token"
fi
export ROBOENV_USER="$USER_NAME"
export ROBOENV_PASSWORD="$PASS"

PUBLIC_IP="${PUBLIC_IPADDR:-}"
EXT_VAR="VAST_TCP_PORT_${PORT}"
EXT_PORT="${!EXT_VAR:-}"

echo "============================================================"
echo " robo-env Gradio"
if [[ "$SCENE" -eq 1 ]]; then
  echo " mode: scene editor (no EO-1 weights)"
else
  echo " mode: playground"
fi
echo
echo " Local:   http://127.0.0.1:${PORT}"
if [[ -n "$PUBLIC_IP" && -n "$EXT_PORT" ]]; then
  echo " Public:  http://${PUBLIC_IP}:${EXT_PORT}"
  echo " HTTPS:   https://${PUBLIC_IP}:${EXT_PORT}  (if vast.ai Instance Portal TLS is on)"
elif [[ -n "$PUBLIC_IP" ]]; then
  echo " Public IP is ${PUBLIC_IP}."
  echo " Mapped port unknown (no ${EXT_VAR})."
  echo " Open the instance card → IP Port Info and use the external port for internal ${PORT}."
else
  echo " Not a vast.ai env (no PUBLIC_IPADDR)."
  echo " On vast.ai: instance card → IP Port Info → external port for internal ${PORT}."
fi
echo
echo " Login username: ${USER_NAME}"
echo " Login token:    ${PASS}"
echo " Token source:   ${PASS_SRC}"
echo
echo " Recover later:"
echo "   echo \$OPEN_BUTTON_TOKEN"
echo "   echo \$JUPYTER_TOKEN"
echo "   cat ${HERE}/.gradio_token"
echo
echo " Optional: set OPEN_BUTTON_PORT=${PORT} on the instance so the Open"
echo " button hits this app. Do not use Gradio share links."
echo " python:    ${VENV_PY}"
echo "============================================================"

if [[ "$SCENE" -eq 1 ]]; then
  exec "$VENV_PY" -m roboenv.ui.scene_app
fi
exec "$VENV_PY" -m roboenv.ui.app
