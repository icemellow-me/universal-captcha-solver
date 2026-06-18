FROM solver-base

# Extra deps for universal solver
RUN pip install --no-cache-dir hcaptcha-challenger 2>/dev/null || true

COPY . /app/

ENV API_KEY=change_me
EXPOSE 8855

CMD ["python3", "solver-server.py", "--api-key", "8010000000ccojr5nrbg516w5jvw1wu9", "--port", "8855"]
