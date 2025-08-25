# Use a slim Python 3.9 image as a base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file to leverage Docker's layer caching
COPY requirements.txt requirements.txt

# Install Dependencies
RUN pip install --no-cahce-dir -r requirements.txt

# Copy application code
COPY . .

# Set up environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose the default port for Streamlit
EXPOSE 8501

# Define the entrypoint command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]