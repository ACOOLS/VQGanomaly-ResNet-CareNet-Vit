# VASTAI
```bash
wget -O ".tmux_conf" https://raw.githubusercontent.com/belarbi2733/dotfiles/master/tmux.conf
tmux source-file .tmux_conf
```
```bash
git clone --branch paperspace https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit.git
cd VQGanomaly-ResNet-CareNet-Vit
wget https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/screw/screw_last_version.zip && unzip screw_last_version.zip && rm screw_last_version.zip
wget https://github.com/ACOOLS/VQGanomaly-ResNet-CareNet-Vit/releases/download/brain/brain_mri_last_version.zip && unzip brain_mri_last_version.zip && rm brain_mri_last_version.zip
```
```bash
apt install python3.10-venv -y 
python -m venv myenv
source myenv/bin/activate
pip install    pytorch-lightning==1.0.8 \
                omegaconf==2.0.0 \
                albumentations==0.4.3 \
                opencv-python==4.5.5.64 \
                pudb==2019.2 \
                imageio==2.9.0 \
                imageio-ffmpeg==0.4.2 \
                torchmetrics==0.4.0 \
                "test-tube>=0.7.5" \
                "streamlit>=0.73.1" \
                einops==0.3.0 \
                torch-fidelity==0.3.0 \
                wandb
export WANDB_API_KEY=cab75a759f850c41f43a9ee4951f98aa6f4a1863
apt install -y libgl1-mesa-glx
pip install --upgrade opencv-python
```
```bash
python main.py --base configs/custom_vqgan_1CH_screw_classique_vastai.yaml -t --gpus 0,
```
