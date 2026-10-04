# Deployment: Linode + Cloudflare + Nginx

This puts the simulation dashboard (`src/web/index.html`) live at **https://amr-edge.zolilabs.com**.

> **Use `amr-edge`, not `amr_edge`, for the hostname.** Underscores aren't valid in hostnames
> (RFC 952/1123), and since CA/Browser Forum Ballot SC-12 certificate authorities can't issue
> certificates that name them. Browser and HTTP-client behavior is also inconsistent. The Nginx
> config accepts both names and redirects `amr_edge` → `amr-edge`, but put the hyphenated name
> in the DNS record and the pitch. The *directory* `/var/www/amr_edge` is fine as it is.

---

## 1. Server diagnostics (Task 1)

Copy the script up and run it:

```bash
scp deploy/linode_diagnostics.sh user@LINODE_IP:~
ssh user@LINODE_IP 'bash ~/linode_diagnostics.sh'
```

Or run the key checks by hand:

```bash
ss -tlnp | grep -E ':80 |:443 '                          # who owns the web ports
systemctl is-active nginx apache2 httpd caddy 2>/dev/null
sudo nginx -T 2>/dev/null | grep server_name | sort -u   # sites already hosted
docker ps --format 'table {{.Names}}\t{{.Ports}}'        # containers binding 80/443?
sudo ufw status
sudo mkdir -p /var/www/amr_edge/releases && sudo chown -R $USER:www-data /var/www/amr_edge
```

**How to read the result:**

| What owns 80/443 | What to do |
|---|---|
| Nothing | `sudo apt update && sudo apt install -y nginx`, then go to step 3 |
| Nginx (host) | Add the site as a new server block (step 3). Existing sites are untouched. |
| Apache | Add a `VirtualHost` with the same root, or move Apache to 8080 behind Nginx |
| Docker proxy (Traefik / nginx-proxy / Caddy) | Mount `/var/www/amr_edge/current` into a static container and route the hostname in that proxy. Don't start a second server on 80/443. |

Open the firewall if needed: `sudo ufw allow 'Nginx Full'`.

## 2. Cloudflare

1. **DNS → Add record**: `A`, name `amr-edge`, IPv4 = Linode IP, **Proxied (orange cloud)**.
2. **SSL/TLS → Overview**: set the mode to **Full (strict)**.
3. **SSL/TLS → Origin Server → Create Certificate**: RSA, hostnames `zolilabs.com, *.zolilabs.com`, 15 years.
   Save both parts on the server:
   ```bash
   sudo mkdir -p /etc/ssl/cloudflare
   sudo nano /etc/ssl/cloudflare/zolilabs.com.pem   # paste the Origin Certificate
   sudo nano /etc/ssl/cloudflare/zolilabs.com.key   # paste the Private Key
   sudo chmod 600 /etc/ssl/cloudflare/zolilabs.com.key
   ```
4. **SSL/TLS → Edge Certificates**: turn on *Always Use HTTPS* and set *Minimum TLS* to 1.2.

The Origin CA certificate is trusted only by Cloudflare, which is exactly right for a proxied
record. If you ever switch the record to DNS-only (grey cloud), use Let's Encrypt instead.

## 3. Nginx

```bash
scp deploy/nginx/amr-edge.zolilabs.com.conf user@LINODE_IP:~
ssh user@LINODE_IP '
  sudo mv ~/amr-edge.zolilabs.com.conf /etc/nginx/sites-available/ &&
  sudo ln -sfn /etc/nginx/sites-available/amr-edge.zolilabs.com.conf /etc/nginx/sites-enabled/ &&
  sudo nginx -t && sudo systemctl reload nginx'
```

On Nginx older than 1.25.1, replace `listen 443 ssl;` + `http2 on;` with `listen 443 ssl http2;`
(check with `nginx -v`).

## 4. Deploy (Task 4)

**Atomic deploy** with timestamped releases and instant rollback, run from the repo root (Git Bash on Windows works):

```bash
DEPLOY_HOST=user@LINODE_IP ./deploy/deploy.sh
```

**One-liner** (no script):

```bash
tar -C src/web -czf - . | ssh user@LINODE_IP 'R=/var/www/amr_edge/releases/$(date +%s); mkdir -p $R && tar -C $R -xzf - && ln -sfn $R /var/www/amr_edge/current && sudo nginx -t && sudo systemctl reload nginx'
```

## 5. Verify

```bash
curl -sI https://amr-edge.zolilabs.com | head -5     # expect HTTP/2 200, server: cloudflare
curl -sI http://amr_edge.zolilabs.com | grep -i location
```

Then purge the Cloudflare cache (**Caching → Purge Everything**) after a redeploy if you changed
the page and Cloudflare is caching HTML.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Cloudflare error **526** | Origin certificate missing or wrong path; the mode is Full (strict) but Nginx is serving a self-signed or default certificate |
| Cloudflare error **521** | Nginx not listening on 443, or the firewall is blocking it |
| **Too many redirects** | SSL mode set to *Flexible*. Change it to *Full (strict)*. |
| Page renders unstyled | The CSP is blocking a CDN. Check the browser console and add the origin to `Content-Security-Policy`. |
