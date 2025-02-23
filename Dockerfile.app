FROM python:3.12-slim

# Set the working directory
WORKDIR /portaai

# Install the dependencies
COPY requirements.txt /portaai
RUN pip install --no-cache-dir -r requirements.txt

COPY service_entrypoint.sh /portaai
COPY doorman.modelfile /portaai/
COPY src /portaai/src

# Run the application
RUN chmod +x service_entrypoint.sh

# Expose the port
EXPOSE 5000

ENTRYPOINT [ "./service_entrypoint.sh" ]