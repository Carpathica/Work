from datetime import datetime, timezone
from typing import Annotated, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from common import auth
from common.dependencies import (
    get_minio_service,
    get_processing_request_service,
    get_yolo_recognizer_service,
    get_yolo_scenario_service,
)
from components.services.minio import MinioService
from components.services.processing_request import ProcessingRequestService
from components.services.yolo_recognizer import YoloRecognizerService
from components.services.yolo_scenario import YoloScenarioService
from schemas.processing_request import (
    ProcessingRequestCreateSchema,
    ProcessingRequestStageSchema,
)
from schemas.yolo_scenario import (
    YoloScenarioCreateSchema,
    YoloScenarioParamsSchema,
    YoloScenarioSchema,
    YoloScenarioUpdateSchema,
)


router = APIRouter(prefix="/api/v1", tags=["YOLO recognition"])


@router.get("/yolo-scenarios", response_model=list[YoloScenarioSchema])
def list_yolo_scenarios(
    name: str = None,
    is_active: bool = None,
    service: YoloScenarioService = Depends(get_yolo_scenario_service),
    current_user=Depends(auth.require_role("admin")),
):
    return service.list(name=name, is_active=is_active)


@router.post("/yolo-scenarios", response_model=YoloScenarioSchema)
def create_yolo_scenario(
    data: YoloScenarioCreateSchema,
    service: YoloScenarioService = Depends(get_yolo_scenario_service),
    current_user=Depends(auth.require_role("admin")),
):
    try:
        return service.create(data, user_id=current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/yolo-scenarios/{scenario_id}", response_model=YoloScenarioSchema)
def update_yolo_scenario(
    scenario_id: UUID,
    data: YoloScenarioUpdateSchema,
    service: YoloScenarioService = Depends(get_yolo_scenario_service),
    current_user=Depends(auth.require_role("admin")),
):
    try:
        return service.update(scenario_id, data, user_id=current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/yolo-scenarios/{scenario_id}/activate", response_model=YoloScenarioSchema)
def activate_yolo_scenario(
    scenario_id: UUID,
    service: YoloScenarioService = Depends(get_yolo_scenario_service),
    current_user=Depends(auth.require_role("admin")),
):
    try:
        return service.activate(scenario_id, user_id=current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/yolo-scenarios/{scenario_id}")
def delete_yolo_scenario(
    scenario_id: UUID,
    service: YoloScenarioService = Depends(get_yolo_scenario_service),
    current_user=Depends(auth.require_role("admin")),
):
    try:
        service.delete(scenario_id, user_id=current_user.id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/yolo-recognize-by-scenario")
async def recognize_yolo_by_scenario(
    files: Annotated[List[UploadFile], File(description="Файлы изображений, 1-2 шт.")],
    scenario: Annotated[str, Form(description="Имя активного YOLO/CNN сценария")],
    yolo_scenario_service: Annotated[
        YoloScenarioService,
        Depends(get_yolo_scenario_service),
    ],
    yolo_recognizer_service: Annotated[
        YoloRecognizerService,
        Depends(get_yolo_recognizer_service),
    ],
    processing_request_service: Annotated[
        ProcessingRequestService,
        Depends(get_processing_request_service),
    ],
    minio_service: Annotated[MinioService, Depends(get_minio_service)],
    current_user=Depends(auth.get_current_user),
    http_request: Request = None,
):
    try:
        yolo_scenario = yolo_scenario_service.get_active(scenario)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    params = YoloScenarioParamsSchema.model_validate(yolo_scenario.params).model_dump(
        mode="json"
    )

    client_ip = http_request.client.host if http_request else None
    return await _process_yolo_request(
        files=files,
        script=yolo_scenario.script,
        model=yolo_scenario.model,
        params=params,
        yolo_recognizer_service=yolo_recognizer_service,
        processing_request_service=processing_request_service,
        minio_service=minio_service,
        current_user=current_user,
        client_ip=client_ip,
        scenario_name=yolo_scenario.name,
        scenario_version=yolo_scenario.version,
    )


@router.post("/yolo-recognize-multipart")
async def recognize_yolo_multipart(
    files: Annotated[List[UploadFile], File(description="Файлы изображений, 1-2 шт.")],
    script: Annotated[str, Form(description="read_container или read_container_KP")],
    model: Annotated[str, Form(description="Имя .pt модели YOLO")],
    yolo_recognizer_service: Annotated[
        YoloRecognizerService,
        Depends(get_yolo_recognizer_service),
    ],
    processing_request_service: Annotated[
        ProcessingRequestService,
        Depends(get_processing_request_service),
    ],
    minio_service: Annotated[MinioService, Depends(get_minio_service)],
    conf: Annotated[float, Form(description="Confidence threshold")] = 0.15,
    iou: Annotated[float, Form(description="YOLO NMS IoU threshold")] = 0.45,
    max_det: Annotated[int, Form(description="Maximum detections count")] = 300,
    merge_iou: Annotated[
        Optional[float],
        Form(description="Postprocessing IoU merge threshold; 0 disables merge"),
    ] = 0.35,
    agnostic_nms: Annotated[bool, Form(description="Use class-agnostic NMS")] = False,
    scenario: Annotated[str, Form()] = "direct",
    current_user=Depends(auth.get_current_user),
    http_request: Request = None,
):
    params = YoloScenarioParamsSchema(
        conf=conf,
        iou=iou,
        max_det=max_det,
        merge_iou=merge_iou,
        agnostic_nms=agnostic_nms,
    ).model_dump(mode="json")

    client_ip = http_request.client.host if http_request else None
    return await _process_yolo_request(
        files=files,
        script=script,
        model=model,
        params=params,
        yolo_recognizer_service=yolo_recognizer_service,
        processing_request_service=processing_request_service,
        minio_service=minio_service,
        current_user=current_user,
        client_ip=client_ip,
        scenario_name=scenario,
    )


async def _process_yolo_request(
    files: List[UploadFile],
    script: str,
    model: str,
    params: dict,
    yolo_recognizer_service: YoloRecognizerService,
    processing_request_service: ProcessingRequestService,
    minio_service: MinioService,
    current_user,
    client_ip: Optional[str] = None,
    scenario_name: Optional[str] = None,
    scenario_version: Optional[int] = None,
) -> dict:
    images = await _read_upload_files(files)
    request_id = uuid4()
    started_at = datetime.now(timezone.utc)

    processing_request_service.create(
        ProcessingRequestCreateSchema(
            id=request_id,
            user_id=current_user.id,
            ip=client_ip,
            stages=[
                ProcessingRequestStageSchema(
                    stage="request_created",
                    processing_result={
                        "recognition_type": "yolo",
                        "images_count": len(images),
                        "scenario": scenario_name,
                        "scenario_version": scenario_version,
                        "script": script,
                        "model": model,
                        "params": params,
                    },
                    timestamp=started_at,
                )
            ],
        )
    )

    file_info = []
    for image in images:
        file_id = minio_service.save_file(
            image["content"],
            content_type=image["content_type"] or "application/octet-stream",
            scenario_name=scenario_name,
            filename=image["filename"],
        )
        file_info.append(
            {
                "file_id": file_id,
                "filename": image["filename"],
                "content_type": image["content_type"],
            }
        )

    processing_request_service.append_stage(
        request_id,
        ProcessingRequestStageSchema(
            stage="file_uploaded",
            processing_result={"files": file_info, "file_storage": "minio"},
            timestamp=datetime.now(timezone.utc),
        ),
    )

    try:
        result = yolo_recognizer_service.recognize(
            images=images,
            script=script,
            model=model,
            params=params,
        )
        result["scenario"] = scenario_name
        result["scenario_version"] = scenario_version

        processing_request_service.append_stage(
            request_id,
            ProcessingRequestStageSchema(
                stage="yolo_recognized",
                processing_result=result,
                timestamp=datetime.now(timezone.utc),
            ),
        )

        status = "success"
        error = None
    except Exception as exc:
        result = None
        status = "error"
        error = str(exc)

    finished_at = datetime.now(timezone.utc)
    metrics = {"duration_ms": (finished_at - started_at).total_seconds() * 1000}

    processing_request_service.append_stage(
        request_id,
        ProcessingRequestStageSchema(
            stage="request_processed",
            processing_result=result,
            metrics=metrics,
            timestamp=finished_at,
            extra={"status": status, "error": error} if error else {"status": status},
        ),
    )
    processing_request_service.set_status(
        request_id,
        status,
        finished_at=finished_at,
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    return {"result": result, "request_id": str(request_id)}


async def _read_upload_files(files: List[UploadFile]) -> list[dict]:
    if not files:
        raise HTTPException(status_code=422, detail="At least one image is required")

    if len(files) > 2:
        raise HTTPException(status_code=422, detail="Only 1-2 images are supported")

    images = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=422, detail=f"Empty file: {file.filename}")

        images.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "content": content,
            }
        )

    return images
