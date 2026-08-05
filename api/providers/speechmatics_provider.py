import os
from typing import Optional

from speechmatics.models import ConnectionSettings, BatchTranscriptionConfig, FetchData
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
from requests_toolbelt import MultipartEncoder

from . import APIProvider, PermanentError, register


@register("speechmatics")
class SpeechmaticsProvider(APIProvider):
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
    ) -> str:
        api_key = os.getenv("SPEECHMATICS_API_KEY")
        if not api_key:
            raise ValueError("SPEECHMATICS_API_KEY environment variable not set")

        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2", auth_token=api_key
        )
        with BatchClient(settings) as client:
            config = BatchTranscriptionConfig(
                language=language,
                enable_entities=True,
                operating_point=model_variant,
            )

            job_id = None
            audio_url = None
            try:
                if use_url:
                    audio_url = sample["row"]["audio"][0]["src"]
                    config.fetch_data = FetchData(url=audio_url)
                    multipart_data = MultipartEncoder(
                        fields={"config": config.as_config().encode("utf-8")}
                    )
                    response = client.send_request(
                        "POST",
                        "jobs",
                        data=multipart_data.to_string(),
                        headers={"Content-Type": multipart_data.content_type},
                    )
                    job_id = response.json()["id"]
                else:
                    job_id = client.submit_job(audio_file_path, config)

                transcript = client.wait_for_completion(
                    job_id, transcription_format="txt"
                )
                return transcript
            except HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise ValueError(
                        "Invalid Speechmatics API credentials"
                    ) from e
                elif e.response.status_code == 400:
                    raise ValueError(
                        f"Speechmatics API responded with 400 Bad request: {e.response.text}"
                    )
                raise e
            except Exception as e:
                if job_id is not None:
                    status = client.check_job_status(job_id)
                    if (
                        audio_url is not None
                        and "job" in status
                        and "errors" in status["job"]
                        and isinstance(status["job"]["errors"], list)
                        and len(status["job"]["errors"]) > 0
                    ):
                        errors = status["job"]["errors"]
                        if "message" in errors[-1] and "failed to fetch file" in errors[-1]["message"]:
                            raise PermanentError(f"could not fetch URL {audio_url}, not retrying") from e

                raise Exception(
                    f"Speechmatics transcription failed: {str(e)}"
                ) from e
