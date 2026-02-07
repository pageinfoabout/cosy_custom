import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

""" CosyVoice Usage, check https://fun-audio-llm.github.io/ for more details
"""
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')
# zero_shot usage
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav')):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cross_lingual usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.',
                                                        './asset/cross_lingual_prompt.wav')):
    torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# vc usage
for i, j in enumerate(cosyvoice.inference_vc('./asset/cross_lingual_prompt.wav', './asset/zero_shot_prompt.wav')):
    torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)




def main():
    # cosyvoice_example()
    # cosyvoice2_example()
    cosyvoice()


if __name__ == '__main__':
    main()
