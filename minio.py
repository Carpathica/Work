import io
import re
from minio import Minio
import uuid

class MinioService:
    def __init__(self, config):
        self.client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
        self.bucket = config.bucket
        
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def save_file(
        self,
        data: bytes,
        content_type: str = "application/octet-stream",
        scenario_name: str | None = None,
        request_id: str | None = None,
        filename: str | None = None,
    ) -> str:
        file_id = str(uuid.uuid4())

        parts = []
        if scenario_name:
            safe_scenario = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_name.strip())
            safe_scenario = safe_scenario.strip("._-")
            if not safe_scenario:
                raise ValueError("Scenario name must contain letters or digits")
            parts.append(safe_scenario[:128])
        if request_id:
            parts.append(request_id)
        if filename and "." in filename:
            extension = filename.rsplit(".", 1)[-1].lower()
            parts.append(f"{file_id}.{extension}")
        elif content_type.startswith("image/"):
            extension = content_type.split("/", 1)[1].split(";", 1)[0].lower()
            parts.append(f"{file_id}.{extension}")
        else:
            parts.append(file_id)
        object_name = "/".join(parts)
        
        if isinstance(data, bytes):
            data = io.BytesIO(data)  # Превращаем в file-like, если надо
            
        self.client.put_object(
            self.bucket,
            object_name,
            data,
            length=len(data.getvalue()),
            content_type=content_type,
        )
        return object_name

    def get_file(self, file_id: str) -> bytes:
        response = self.client.get_object(self.bucket, file_id)
        data = response.read()
        response.close()
        return data
