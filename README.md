# alera 

<div style="text-align: center;">
  <img src="src/app/static/images/logo.jpeg" alt="Logo" width="300" style="margin: auto;"/>
</div>

We are developing a virtual doorman with artificial intelligence that replaces the human doorman, enabling audio communication with visitors, facial recognition for residents, and authorization via a mobile app. It also detects suspicious behavior and sends security alerts.

## Checklist

### 0) Literature Seach
- [x] Find appropriate models for the project.

### 1) LLM Model
- [x] Comparison between LLM models.
- [x] Create a modelfile.
- [ ] Parameters.
- [ ] Fine Tunning.

### 2) Face Detection + Recognition -> Attendence
- [x] Face Detector.
- [x] Register Users based on Face.
- [x] Classification based on Face.

### 3) Speech-to-Text
- [ ] Comparison between models.
- [ ] Train or adapt a model for our purposes.
- [ ] Real-Time.

### 4) Text-to-Speech 
- [ ] Comparison between models.
- [ ] Train or adapt a model for our purposes.
- [ ] Real-Time.
- [ ] Convert into packages. 

### 5) Hardware Integration
- [ ] esp32 collecting (microfone) and receiving (internet) audio packages from server in real time.
- [ ] TDP (the Tag Distribution Protocol).

### 6) Mobile Application
- [ ] Integrate all functionalities with a mobile app.


## Run the app
```
docker compose up
```
or 
```
pip install -r requirements 
```
and run the main.py file.   

## Docker installation on Ubuntu 22.04

### Docker
<a href="https://docs.docker.com/engine/install/ubuntu/">https://docs.docker.com/engine/install/ubuntu/</a>
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
and then
```
 sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Docker GPU support 
<a href="https://runs-on.com/blog/3-how-to-setup-docker-with-nvidia-gpu-support-on-ubuntu-22/">How to setup docker with NVIDIA GPU support on Ubuntu 22</a>
```
cd /tmp/
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers-545 nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### NVIDIA Container Toolkit
<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">Installing the NVIDIA Container Toolkit</a>
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
then (optionally)
```
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Update Package
```
sudo apt-get update
```
Install the NVIDIA Container Toolkit packages:
```
sudo apt-get install -y nvidia-container-toolkit
```

## Troubleshouting
See all containers
```
docker ps
```
See all images
```
docker images
```
See all volumes
```
docker volume ls
```
Remove containers and images
```
docker compose down --rmi all
```
Remove all volumes
```
docker volume prune --all
```
clean up / identify contents of ```/var/lib/docker/overlay``` (docker storage driver)
```
du -ahx / | sort -rh | head -50
```
```
ls /var/lib/docker/overlay2
```
remove all contents in storage driver
```
docker buildx prune --all
```
removal everything
```
docker system prune -a
```


## Models
go to <a href="https://ollama.com/library">Ollama Library</a> an pick the most suitable model.