from pathlib import Path

import PIL.Image
import torch
import torchvision
import transformers
import ttnn
from loguru import logger
from models.experimental.table_transformer.tt.table_transformer import TableTransformer
from models.utility_functions import comp_allclose, comp_pcc

SCRIPT_DIR = Path(__file__).parent


def preprocess_images(images: [PIL.Image.Image]) -> torch.Tensor:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return torch.stack([transform(image) for image in images])


def test_table_transformer_detection(device: ttnn.Device, pcc: float = 0.999) -> None:
    hf_model = transformers.TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )
    hf_model.eval()

    tt_model = TableTransformer(device=device, num_classes=2, num_queries=15)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth",
        map_location="cpu",
        check_hash=True,
    )

    for k in list(state_dict.keys()):
        if k.startswith("backbone.0.body."):
            k_new = k.replace("backbone.0.body.", "backbone.")
            state_dict[k_new] = state_dict.pop(k)

    tt_model.load_state_dict(state_dict)
    tt_model.eval()

    image_paths = ["../images/PMC1079941_2.jpg", "../images/PMC1079941_3.jpg"]
    pil_images = [PIL.Image.open(SCRIPT_DIR / path) for path in image_paths]
    torch_images = preprocess_images(pil_images)

    with torch.no_grad():
        hf_output = hf_model.forward(torch_images)

    tt_output = tt_model(torch_images)

    hf_logits = hf_output["logits"]
    hf_boxes = hf_output["pred_boxes"]

    tt_logits = ttnn.to_torch(tt_output["logits"])
    tt_boxes = ttnn.to_torch(tt_output["boxes"])

    _passing, comp_output = comp_allclose(hf_logits, tt_logits, pcc)
    logger.info(f"Logits - {comp_output}")

    _passing, comp_output = comp_allclose(hf_boxes, tt_boxes, pcc)
    logger.info(f"Boxes - {comp_output}")

    passing, comp_output = comp_pcc(hf_logits, tt_logits, pcc)
    logger.info(f"Logits PCC: {comp_output}")
    assert passing, f"Model output `logits` does not meet PCC requirement {pcc}."

    passing, comp_output = comp_pcc(hf_boxes, tt_boxes, pcc)
    logger.info(f"Boxes PCC: {comp_output}")
    assert passing, f"Model output `boxes` does not meet PCC requirement {pcc}."


# def test_table_transformer_recognition(device, pcc=0.95):
#     hf_model = transformers.TableTransformerForObjectDetection.from_pretrained(
#         "microsoft/table-transformer-structure-recognition-v1.1-all"
#     )
