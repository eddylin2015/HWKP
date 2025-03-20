# WSL+ollama+deepseek-r1:8b

你也可以直接使用 Windows + R 快捷键，在打开的「运行」窗口中直接执行 optionalfeatures 打开「Windows 功能」对话框。

Windows 虚拟化
WSL2
Hyper-v

bcdedit /set hypervisorlauchtype auto

开启hyper-v

>wsl --install

>wsl --update

>wsl --install -d Ubuntu

user:

password:

wsl>exit

>wsl --export Ubuntu D:\backup\ubuntu.tar

>wsl --unregister Ubuntu

>wsl --import Ubuntu D:\wsl\ D:\backup\ubuntu.tar

>wsl

wsl>curl -fsSL https://ollama.com/install.sh | sh

wsl>ollama serve

wsl>ollama run deepseek-r1:8b

# AMD CPU Ryzen AI Max 395 pro for deepseek-r1:70b

https://community.amd.com/t5/ai/amd-ryzen-ai-max-395-processor-breakthrough-ai-performance-in/ba-p/752960

# deepseek-coder:1.3b 6.7b 33b
ollama run deepseek-coder
ollama run deepseek-coder:6.7b
ollama run deepseek-coder:33b

curl -X POST http://localhost:11434/api/generate -d '{
  "model": "deepseek-coder",
  "prompt":"Why is the sky blue?"
 }'

# GTC March 2025 Keynote with NVIDIA CEO Jensen Huang

https://www.youtube.com/watch?v=_waPvOwL9Z8



