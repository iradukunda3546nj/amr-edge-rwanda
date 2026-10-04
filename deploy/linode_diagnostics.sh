#!/usr/bin/env bash
# AMR-Edge Rwanda: Linode server diagnostics + deployment directory setup.
# Read-only checks first; the only writes are in section 6 (directory setup).
# Usage on the server:  bash linode_diagnostics.sh
set -uo pipefail

hr() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

hr "1. System"
uname -a
. /etc/os-release && echo "OS: $PRETTY_NAME"
uptime
free -h
df -h /

hr "2. Public IP (must match the Cloudflare A record)"
curl -4 -s --max-time 5 https://ifconfig.me || echo "could not resolve public IP"
echo

hr "3. Who is listening on 80 / 443?"
sudo ss -tlnp | awk 'NR==1 || /:80 |:443 /'

hr "4. Web servers installed / running"
for svc in nginx apache2 httpd caddy lighttpd; do
  if systemctl list-unit-files 2>/dev/null | grep -q "^${svc}.service"; then
    printf '%-10s %s\n' "$svc" "$(systemctl is-active "$svc")"
  fi
done
command -v nginx >/dev/null && nginx -v 2>&1
if command -v nginx >/dev/null; then
  echo "--- server_name entries already configured:"
  sudo nginx -T 2>/dev/null | grep -E '^\s*server_name' | sort -u
fi

hr "5. Docker"
if command -v docker >/dev/null; then
  sudo docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
else
  echo "docker not installed"
fi

hr "5b. Firewall"
sudo ufw status 2>/dev/null || sudo iptables -L INPUT -n --line-numbers 2>/dev/null | head -20

hr "6. Deployment directory"
sudo mkdir -p /var/www/amr_edge/releases
sudo chown -R "$USER":www-data /var/www/amr_edge
sudo chmod -R 2775 /var/www/amr_edge
ls -la /var/www/amr_edge

hr "Done"
echo "If port 80/443 is owned by Docker (e.g. a reverse-proxy container), add the"
echo "site to that proxy instead of installing a second Nginx. See docs/DEPLOYMENT.md."
