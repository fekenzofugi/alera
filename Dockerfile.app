FROM python:3.12-slim

# Set the working directory
WORKDIR /portaai

# Install the dependencies
RUN apt-get update && apt-get install -y espeak ffmpeg cmake build-essential libgtk-3-dev libboost-all-dev
COPY requirements.txt /portaai
RUN pip install -U pip wheel cmake
RUN pip install --no-cache-dir -r requirements.txt

COPY service_entrypoint.sh /portaai
COPY doorman.modelfile /portaai/
COPY . /portaai/

# Run the application
RUN chmod +x service_entrypoint.sh

# Expose the port
EXPOSE 5000

ENTRYPOINT [ "./service_entrypoint.sh" ]