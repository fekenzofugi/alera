from gtts import gTTS

def text_to_speech_v1(text, outpath, chat_title):
    tts = gTTS(text=text, lang='pt-br')
    tts.save(f'{outpath}/{chat_title}.mp3')