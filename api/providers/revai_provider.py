import os
import time
from typing import Optional

from rev_ai import apiclient
from rev_ai.models import CustomerUrlData

from . import APIProvider, register


@register("revai")
class RevAIProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        access_token = os.getenv("REVAI_API_KEY")
        client = apiclient.RevAiAPIClient(access_token)

        if use_url:
            job = client.submit_job_url(
                transcriber=model_variant,
                source_config=CustomerUrlData(sample["row"]["audio"][0]["src"]),
                metadata="benchmarking_job",
            )
        else:
            job = client.submit_job_local_file(
                transcriber=model_variant,
                filename=audio_file_path,
                metadata="benchmarking_job",
            )

        # Polling until job is done
        while True:
            job_details = client.get_job_details(job.id)
            if job_details.status.name in ["IN_PROGRESS", "TRANSCRIBING"]:
                time.sleep(0.1)
                continue
            elif job_details.status.name == "FAILED":
                raise Exception("RevAI transcription failed.")
            elif job_details.status.name == "TRANSCRIBED":
                break

        transcript_object = client.get_transcript_object(job.id)

        # Combine all words from all monologues
        transcript_text = []
        for monologue in transcript_object.monologues:
            for element in monologue.elements:
                transcript_text.append(element.value)

        return "".join(transcript_text) if transcript_text else ""
