import base64
from datetime import datetime, timezone
import io
from typing import Annotated, List, Optional
from uuid import uuid4
from PIL import Image
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from common import auth
from common.dependencies import get_minio_service, get_processing_request_service, get_scenario_router_service
from components.services.minio import MinioService
from components.services.processing_request import ProcessingRequestService
from components.services.scenario_router import ScenarioRouterService
from schemas.processing_request import ProcessingRequestCreateSchema, ProcessingRequestStageSchema
from schemas.scenarios.scenario_router import RecognizeMetadata, RecognizeRequest

router = APIRouter(prefix="/api/v1", tags=["Scenario processing"])



@router.post("/recognize")
async def recognize(
    request: RecognizeRequest,
    scenario_router: Annotated[ScenarioRouterService, Depends(get_scenario_router_service)],
    processing_request_service: Annotated[ProcessingRequestService, Depends(get_processing_request_service)],
    minio_service: Annotated[MinioService, Depends(get_minio_service)],
    current_user = Depends(auth.get_current_user),
    http_request: Request = None,
):
    """
    Endpoint для распознавания по сценарию.
    """
    client_ip = http_request.client.host if http_request else None
    return await process_recognition_request(
        request=request,
        scenario_router=scenario_router,
        processing_request_service=processing_request_service,
        minio_service=minio_service,
        current_user=current_user,
        client_ip=client_ip,
        scenario_name=request.metadata.scenario,
    )


@router.post("/recognize-multipart")
async def recognize_multipart(
    scenario_router: Annotated[ScenarioRouterService, Depends(get_scenario_router_service)],
    processing_request_service: Annotated[ProcessingRequestService, Depends(get_processing_request_service)],
    minio_service: Annotated[MinioService, Depends(get_minio_service)],
    files: Annotated[Optional[List[UploadFile]], File(description="Файлы изображений (jpeg/png)")],
    metadata: Annotated[str, Form(description="Метаданные в формате JSON (строка)", example='{"scenario": "container_site"}')] = '{"scenario": "extract_number_container_site"}',
    current_user=Depends(auth.get_current_user),
    http_request: Request = None,
):
    """
    Multipart endpoint для распознавания по сценарию (Swagger-friendly).
    """
    # 1. Конвертируем файлы в base64
    images_b64: List[str] = []
    if files:
        for file in files:
            content = await file.read()
            images_b64.append(base64.b64encode(content).decode())

    # 2. Парсим metadata
    try:
        metadata_obj = RecognizeMetadata.model_validate_json(metadata)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid metadata JSON: {e}")

    # 3. Собираем pydantic-запрос для повторного использования логики
    recognize_request = RecognizeRequest(
        images=images_b64,
        metadata=metadata_obj
    )

    client_ip = http_request.client.host if http_request else None
    return await process_recognition_request(
        request=recognize_request,
        scenario_router=scenario_router,
        processing_request_service=processing_request_service,
        minio_service=minio_service,
        current_user=current_user,
        client_ip=client_ip,
        scenario_name=metadata_obj.scenario,
    )


async def process_recognition_request(
    request: RecognizeRequest,
    scenario_router: ScenarioRouterService,
    processing_request_service: ProcessingRequestService,
    minio_service: MinioService,
    current_user,
    client_ip: Optional[str] = None,
    scenario_name: Optional[str] = None,
) -> dict:
    """
    Унифицированная логика обработки запроса (base64/images), с логированием, метриками и MinIO.
    Используется обоими эндпоинтами.
    """
    request_id = uuid4()
    started_at = datetime.now(timezone.utc)
    
    # 1. Формируем заявку с первым этапом
    stage = ProcessingRequestStageSchema(
        stage="request_created",
        processing_result={"images_count": len(request.images)},
        metrics=None,
        timestamp=started_at,
        extra={"metadata": request.metadata}
    )
    processing_request_service.create(
        ProcessingRequestCreateSchema(
            id=request_id,
            scenario_id=getattr(request.metadata, "scenario_id", None),
            user_id=current_user.id,
            ip=client_ip,
            stages=[stage]
        )
    )
    
    # 2. Сохраняем исходные изображения в minio
    file_info = []
    images_for_router = []
    for img_b64 in request.images:
        img_bytes = base64.b64decode(img_b64)
        try:
            img = Image.open(io.BytesIO(img_bytes))
            fmt = img.format.lower()
            images_for_router.append(img)
            content_type = f"image/{fmt}"
        except Exception:
            content_type = None
        file_id = minio_service.save_file(
            img_bytes,
            content_type=content_type or "application/octet-stream",
            scenario_name=scenario_name,
            request_id=str(request_id),
        )
        file_info.append({"file_id": file_id, "content_type": content_type})
        
    # 3. Этап "file_uploaded" с file_ids
    input_stage = ProcessingRequestStageSchema(
        stage="file_uploaded",
        timestamp=datetime.now(timezone.utc),
        processing_result={"files": file_info, "file_storage": "minio"}
    )
    processing_request_service.append_stage(request_id, input_stage)

    # 4. Основная обработка
    try:
        result = await scenario_router.run(
            request_id=request_id,
            scenario_name=request.metadata.scenario,
            images=images_for_router,
            metadata=request.metadata
        )
        status = "success"
        error = None
    except Exception as e:
        result = None
        status = "error"
        error = str(e)
    
    # 5. Логируем метрики и финальный этап
    finished_at = datetime.now(timezone.utc)
    metrics = {"duration_ms": (finished_at - started_at).total_seconds() * 1000}
    processing_request_service.append_stage(
        request_id,
        ProcessingRequestStageSchema(
            stage="request_processed",
            timestamp=finished_at,
            metrics=metrics,
            processing_result=result,
            extra={"status": status, "error": error} if error else {"status": status}
        )
    )
    
    # Смена статуса
    processing_request_service.set_status(
        request_id,
        status,
        finished_at=finished_at,
    )

    if error:
        raise HTTPException(status_code=400, detail=error)
    return {"result": result, "request_id": str(request_id)}
