#!/usr/bin/env bash
# Mini-WA-JEPA 停止服务脚本
pkill -f "marimo run app.py" || true
pkill -f "marimo edit app.py" || true
which notify-send >/dev/null 2>&1 && notify-send "Mini-WA-JEPA" "服务已停止" || true
echo "[✓] Mini-WA-JEPA 服务已成功停止。"
