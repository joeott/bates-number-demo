# Multi-stage Dockerfile for Legal Retriever UI

# Stage 1: Backend build
FROM python:3.9-slim as backend-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-dotenv

# Stage 2: Frontend build
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY ui/frontend/package*.json ./
RUN npm ci --only=production || echo "No package.json found"

COPY ui/frontend/ ./

# Optional: Minify assets
RUN npx terser script.js -c -m -o script.min.js || cp script.js script.min.js
RUN npx csso style.css -o style.min.css || cp style.css style.min.css

# Stage 3: Final runtime image
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=backend-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Copy optimized frontend from builder
COPY --from=frontend-builder /app/frontend /app/ui/frontend

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Nginx configuration
RUN echo 'server { \n\
    listen 80; \n\
    server_name localhost; \n\
    \n\
    location / { \n\
        root /app/ui/frontend; \n\
        try_files $uri $uri/ /index.html; \n\
    } \n\
    \n\
    location /api { \n\
        proxy_pass http://localhost:8080; \n\
        proxy_http_version 1.1; \n\
        proxy_set_header Upgrade $http_upgrade; \n\
        proxy_set_header Connection "upgrade"; \n\
        proxy_set_header Host $host; \n\
        proxy_set_header X-Real-IP $remote_addr; \n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \n\
        proxy_set_header X-Forwarded-Proto $scheme; \n\
    } \n\
    \n\
    location /health { \n\
        proxy_pass http://localhost:8080/health; \n\
    } \n\
}' > /etc/nginx/sites-available/default

# Supervisor configuration
RUN echo '[supervisord] \n\
nodaemon=true \n\
user=root \n\
\n\
[program:api] \n\
command=python -m uvicorn ui.backend.api:app --host 0.0.0.0 --port 8080 \n\
directory=/app \n\
user=appuser \n\
autostart=true \n\
autorestart=true \n\
redirect_stderr=true \n\
stdout_logfile=/dev/stdout \n\
stdout_logfile_maxbytes=0 \n\
environment=PYTHONUNBUFFERED="1" \n\
\n\
[program:nginx] \n\
command=/usr/sbin/nginx -g "daemon off;" \n\
autostart=true \n\
autorestart=true \n\
redirect_stderr=true \n\
stdout_logfile=/dev/stdout \n\
stdout_logfile_maxbytes=0' > /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 80
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]