#!/usr/bin/env bash
# Build everything and install the KnctXplore launcher (app-menu entry + `knctxplore` command).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
cargo build --release
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/scalable/apps"
ln -sf "$here/target/release/knctxplore" "$HOME/.local/bin/knctxplore"
cp assets/knctxplore.svg "$HOME/.local/share/icons/hicolor/scalable/apps/knctxplore.svg"
cat > "$HOME/.local/share/applications/knctxplore.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=KnctXplore
Comment=Xbox 360 Kinect: live viewer, 3D body and bust scan booths
Exec=$here/target/release/knctxplore
Icon=knctxplore
Terminal=false
Categories=Graphics;3DGraphics;
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true
echo "Installed. Launch it from the app menu, or run: knctxplore"
