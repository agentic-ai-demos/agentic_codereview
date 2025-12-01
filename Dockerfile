FROM python:3.10-slim

FROM python:3.10-slim

# Install system dependencies for building native extensions
# Create non-root user
RUN useradd -m -u 1000 user

# Environment
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
RUN chown -R user:1000 /app
# Install pip and uv
RUN pip install --upgrade pip && pip install uv

# Copy project files
COPY --chown=user:1000 . /app

# Switch to non-root user
USER user

# Install dependencies
RUN uv sync

# Run the server
CMD ["uv", "run", "uvicorn", "code_review_agent:app", "--host", "0.0.0.0", "--port", "7860"]

# Build the docker
# docker build -t codereview-agent .
# run as
# docker run -it --rm -p 7860:7860 codereview-agent