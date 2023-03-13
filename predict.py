import os
import shutil
import subprocess
import tempfile
import warnings

import torchaudio
from cog import BasePredictor, Input, Path

from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import load_voices

warnings.filterwarnings(
    "ignore"
)  # pysndfile does not support mp3, so we silence its warnings

VOICE_OPTIONS = [
    "angie",
    "cond_latent_example",
    "deniro",
    "freeman",
    "halle",
    "lj",
    "myself",
    "pat2",
    "snakes",
    "tom",
    "train_daws",
    "train_dreams",
    "train_grace",
    "train_lescault",
    "weaver",
    "applejack",
    "daniel",
    "emma",
    "geralt",
    "jlaw",
    "mol",
    "pat",
    "rainbow",
    "tim_reynolds",
    "train_atkins",
    "train_dotrice",
    "train_empire",
    "train_kennard",
    "train_mouse",
    "william",
    "random",  # special option for random voice
    "custom_voice",  # special option for custom voice
    "disabled",  # special option for disabled voice
]

MODULE_DIRECTORY = os.path.dirname(__file__)
CUSTOM_VOICE_DIRECTORY = Path(MODULE_DIRECTORY, "tortoise", "voices", "custom_voice")


def create_custom_voice_from_mp3(input_path: str, segment_time: int = 9) -> None:
    CUSTOM_VOICE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    normalized_audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.check_output(
        [
            "ffmpeg-normalize",
            input_path,
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-o",
            normalized_audio_path,
        ]
    )
    assert os.path.exists(normalized_audio_path), "ffmpeg-normalize failed"
    subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "warning",  # log only errors
            "-ss",
            "00:00:00",
            "-t",
            "300",  # limit to 300 seconds (~30 segments)
            "-i",
            normalized_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            os.path.join(CUSTOM_VOICE_DIRECTORY, "%d.wav"),
        ]
    )


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.text_to_speech = TextToSpeech(
            models_dir=MODELS_DIR, autoregressive_batch_size=16, device="cuda"
        )

    def predict(
        self,
        text: str = Input(
            description="Text to speak.",
            default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.",
        ),
        voice_a: str = Input(
            description="Selects the voice to use for generation. Use `random` to select a random voice. Use `custom_voice` to use a custom voice.",
            default="random",
            choices=VOICE_OPTIONS,
        ),
        custom_voice: Path = Input(
            description="(Optional) Create a custom voice based on an mp3 file of a speaker. Audio should be at least 15 seconds, only contain one speaker, and be in mp3 format. Overrides the `voice_a` input.",
            default=None,
        ),
        voice_b: str = Input(
            description="(Optional) Create new voice from averaging the latents for `voice_a`, `voice_b` and `voice_c`. Use `disabled` to disable voice mixing.",
            default="disabled",
            choices=VOICE_OPTIONS,
        ),
        voice_c: str = Input(
            description="(Optional) Create new voice from averaging the latents for `voice_a`, `voice_b` and `voice_c`. Use `disabled` to disable voice mixing.",
            default="disabled",
            choices=VOICE_OPTIONS,
        ),
        preset: str = Input(
            description="Which voice preset to use. See the documentation for more information.",
            default="fast",
            choices=["ultra_fast", "fast", "standard", "high_quality"],
        ),
        seed: int = Input(
            description="Random seed which can be used to reproduce results.",
            default=0,
        ),
        cvvp_amount: float = Input(
            description="How much the CVVP model should influence the output. Increasing this can in some cases reduce the likelyhood of multiple speakers. Defaults to 0 (disabled)",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
    ) -> Path:
        output_dir = Path(tempfile.mkdtemp())

        if custom_voice is not None:
            assert (
                custom_voice.suffix == ".mp3"
            ), f"File {custom_voice} is not an mp3 file"
            print(f"Creating voice from {custom_voice}")
            # remove the custom voice dir if it exists
            shutil.rmtree(str(CUSTOM_VOICE_DIRECTORY), ignore_errors=True)
            create_custom_voice_from_mp3(str(custom_voice))
            all_voices = ["custom_voice"]
        else:
            all_voices = [voice_a]
        if voice_b != "disabled":
            all_voices.append(voice_b)
        if voice_c != "disabled":
            all_voices.append(voice_c)
        print(f"Generating text using voices: {all_voices}")

        voice_samples, conditioning_latents = load_voices(all_voices)
        generated_speech, _ = self.text_to_speech.tts_with_preset(
            text,
            k=1,  # k=1 means we only generate one candidate
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
            use_deterministic_seed=seed,
            return_deterministic_state=True,
            cvvp_amount=cvvp_amount,
        )

        audio_raw_path = output_dir.joinpath(f"tortoise.wav")
        torchaudio.save(str(audio_raw_path), generated_speech.squeeze(0).cpu(), 24000)

        audio_mp3_path = output_dir.joinpath(f"tortoise.mp3")
        subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(audio_raw_path),
                str(audio_mp3_path),
            ],
        )
        audio_raw_path.unlink()  # Delete the raw audio file
        shutil.rmtree(
            str(CUSTOM_VOICE_DIRECTORY), ignore_errors=True
        )  # Delete the custom voice dir, if it exists
        return audio_mp3_path