import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
print(cosyvoice.list_available_spks())






def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice()


if __name__ == '__main__':
    main()
