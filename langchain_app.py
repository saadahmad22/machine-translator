from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import librosa
import soundfile as sf  # to load audio files

class LangChainApp:
    '''A class to process an audio file containing Spanish speech to English text.'''

    def __init__(self, lang1: str = "en", lang2: str = "es", speech_to_text_model: str = "openai/whisper-base", translation_model: str=None, speech_kwargs: dict=None, translation_model_kwargs: dict=None) -> None:
        '''Initialize the LangChainApp with the specified speech-to-text and translation models.'''

        self.lang1 = lang1
        self.lang2 = lang2
        if not translation_model:
            translation_model = f"Helsinki-NLP/opus-mt-{lang1}-{lang2}"
        # Initialize the Whisper pipeline for Spanish ASR
        speech_processor = AutoProcessor.from_pretrained(speech_to_text_model, clean_up_tokenization_spaces=True)
        speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(speech_to_text_model, use_safetensors=True, **speech_kwargs if speech_kwargs else {})
        self.pipe_whisper = pipeline(
                "automatic-speech-recognition",
                model=speech_model,
                tokenizer=speech_processor.tokenizer,
                feature_extractor=speech_processor.feature_extractor,
                device="mps",
                # chunks of 10 seconds, default stride
                chunk_length_s=10
            )

        # Initialize the translation pipeline for Spanish to English translation
        self.pipe_translation = HuggingFacePipeline(
            pipeline=pipeline(
                "translation",
                model=translation_model,
                device=0 if torch.cuda.is_available() else -1
            )
        )

    @staticmethod
    def preprocess_audio(audio_path: str):
        '''Preprocess the audio file and return the audio tensor.'''

        audio, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='linear')
        return audio
    
    def process_audio_file(self, audio_path: str):
        '''Process the audio file and return the translated text.'''

        # Load the audio file using soundfile
        audio, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000, res_type='linear')

        # Transcribe the audio to Spanish text
        transcription = self.pipe_whisper(
            audio, 
            generate_kwargs={
                "task": "transcribe",
                "language": [self.lang1], 
                "is_multilingual":True,
                "condition_on_prev_tokens": True})["text"]

        # Translate the Spanish text to English text
        translation = self.pipe_translation.invoke(transcription)

        return (transcription, translation)
    
    def translate_text(self, text: str):
        '''Translate the given text from Spanish to English.'''

        return self.pipe_translation.invoke(text)


# Example usage
app = LangChainApp()

# # Replace with the actual path to your audio file (WAV format or supported)
# audio_file_path = "days-of-week.mp3"
translation = app.translate_text("Hello, how is it going?")

print("Translated text:", translation)