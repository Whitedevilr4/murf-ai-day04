from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import List

# backend folder
BASE_DIR = Path(__file__).resolve().parent.parent
ORDERS_DIR = BASE_DIR / "orders"


@dataclass
class OrderState:
    drinkType: str
    size: str
    milk: str
    extras: List[str]
    name: str


def save_order_to_file(order: OrderState) -> str:
    """
    Save a JSON file that summarizes the order,
    and return a human-readable summary string.
    """
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)

    extras_text = "no extras"
    if order.extras:
        extras_text = ", ".join(order.extras)

    summary_text = (
        f"{order.name}'s order: a {order.size} {order.drinkType} "
        f"with {order.milk} and {extras_text}."
    )

    data = {
        "drinkType": order.drinkType,
        "size": order.size,
        "milk": order.milk,
        "extras": order.extras,
        "name": order.name,
        "summary": summary_text,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    filename = ORDERS_DIR / f"order-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    with filename.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return summary_text + " Your drink will be ready shortly."
