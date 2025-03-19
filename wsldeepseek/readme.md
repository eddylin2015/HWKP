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