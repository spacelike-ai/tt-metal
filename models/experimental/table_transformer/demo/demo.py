from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import PIL.Image
import torch
import torchvision
import ttnn
from loguru import logger
from matplotlib import pyplot as plt
from models.experimental.table_transformer.tt.table_transformer import TableTransformer

SCRIPT_DIR = Path(__file__).parent


def box_cxcywh_to_xyxy(x):  # noqa: ANN001, ANN201
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):  # noqa: ANN001, ANN201
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)


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


def postprocess_output(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(-1)[:, :-1]
    keep = probabilities.max(-1).values > 0.7  # noqa: PD011, PLR2004

    bboxes_scaled = rescale_bboxes(boxes[keep], image_size)
    return probabilities[keep], bboxes_scaled


def plot_results(
    pil_img: PIL.Image.Image,
    prob: torch.Tensor,
    boxes: torch.Tensor,
    classes: list[str],
    filepath: Path,
) -> None:
    colors = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors * 100):
        ax.add_patch(mpl.patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f"{classes[cl]}: {p[cl].item():0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox={"facecolor": "yellow", "alpha": 0.5})

    plt.axis("off")
    plt.savefig(filepath)

    logger.info(f"Output image saved as {filepath}")


def test_table_transformer_detection(device: ttnn.Device) -> None:
    model = TableTransformer(device=device, num_classes=2, num_queries=15)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth",
        map_location="cpu",
        check_hash=True,
    )

    for k in list(state_dict.keys()):
        if k.startswith("backbone.0.body."):
            k_new = k.replace("backbone.0.body.", "backbone.")
            state_dict[k_new] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    image_paths = ["../images/PMC1079941_2.jpg", "../images/PMC1079941_3.jpg"]
    pil_images = [PIL.Image.open(SCRIPT_DIR / path) for path in image_paths]
    torch_images = preprocess_images(pil_images)

    output = model(torch_images)
    all_logits = ttnn.to_torch(output["logits"])
    all_boxes = ttnn.to_torch(output["boxes"])

    for i, pil_image in enumerate(pil_images):
        scores, boxes = postprocess_output(all_logits[i], all_boxes[i], pil_image.size)

        classes = ["table", "table (rotated)"]
        plot_results(pil_image, scores, boxes, classes, f"table_transformer_detect_{i}.jpeg")
