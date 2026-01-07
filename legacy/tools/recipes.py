"""
Recipe tools for the voice assistant.
Currently supports pizza dough calculations using baker's percentages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import sys


DEFAULT_CONFIG: Dict[str, Any] = {
    "style": "neapolitan",
    "ball_weight_g": 260.0,
    "ball_count": 4,
    "hydration_percent": 62.0,
    "salt_percent": 2.8,
    "yeast_percent": 0.1,
    "oil_percent": 0.0,
    "cold_ferment_hours": 18.0,
    "room_temp_hours": 2.0,
    "bake_temp_f": 850,
}

_LAST_RECIPE: Optional[Dict[str, Any]] = None


def _get_event_bus():
    """Locate the shared EventBus instance if available."""
    try:
        if "wakeword" in sys.modules and hasattr(sys.modules["wakeword"], "event_bus"):
            return sys.modules["wakeword"].event_bus
        if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "event_bus"):
            return sys.modules["__main__"].event_bus
    except Exception:
        return None
    return None


def _emit_ui_card(card: Dict[str, Any]) -> None:
    """Emit a ui_card event if the event bus is available."""
    bus = _get_event_bus()
    if not bus:
        return
    bus.emit("ui_card", {"card": card, "replace_existing": True})


def _coerce_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")


def _coerce_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")


def _merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _validate_config(config: Dict[str, Any]) -> None:
    if config["ball_count"] <= 0:
        raise ValueError("ball_count must be greater than 0")
    if config["ball_weight_g"] <= 0:
        raise ValueError("ball_weight_g must be greater than 0")
    if not (50 <= config["hydration_percent"] <= 85):
        raise ValueError("hydration_percent must be between 50 and 85")
    if not (0 <= config["salt_percent"] <= 5):
        raise ValueError("salt_percent must be between 0 and 5")
    if not (0 <= config["yeast_percent"] <= 2):
        raise ValueError("yeast_percent must be between 0 and 2")
    if not (0 <= config["oil_percent"] <= 10):
        raise ValueError("oil_percent must be between 0 and 10")
    if config["cold_ferment_hours"] < 0:
        raise ValueError("cold_ferment_hours must be 0 or greater")
    if config["room_temp_hours"] < 0:
        raise ValueError("room_temp_hours must be 0 or greater")
    if not (400 <= config["bake_temp_f"] <= 1000):
        raise ValueError("bake_temp_f must be between 400 and 1000")


def _round_weight(value: float) -> float:
    return round(value, 1)


def _build_ingredients(config: Dict[str, Any]) -> Dict[str, Any]:
    total_dough = config["ball_weight_g"] * config["ball_count"]
    hydration = config["hydration_percent"] / 100.0
    salt = config["salt_percent"] / 100.0
    yeast = config["yeast_percent"] / 100.0
    oil = config["oil_percent"] / 100.0

    total_factor = 1.0 + hydration + salt + yeast + oil
    flour_g = total_dough / total_factor
    water_g = flour_g * hydration
    salt_g = flour_g * salt
    yeast_g = flour_g * yeast
    oil_g = flour_g * oil

    ingredients = [
        {
            "name": "Flour (00)",
            "grams": _round_weight(flour_g),
            "bakers_percent": 100.0,
        },
        {
            "name": "Water",
            "grams": _round_weight(water_g),
            "bakers_percent": config["hydration_percent"],
        },
        {
            "name": "Salt",
            "grams": _round_weight(salt_g),
            "bakers_percent": config["salt_percent"],
        },
        {
            "name": "Yeast (instant)",
            "grams": _round_weight(yeast_g),
            "bakers_percent": config["yeast_percent"],
        },
    ]

    if config["oil_percent"] > 0:
        ingredients.append(
            {
                "name": "Olive oil",
                "grams": _round_weight(oil_g),
                "bakers_percent": config["oil_percent"],
            }
        )

    return {
        "total_dough_g": _round_weight(total_dough),
        "ingredients": ingredients,
    }


def _build_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = [
        {
            "number": 1,
            "instruction": "Mix water and yeast, then add flour and stir until shaggy.",
        },
        {
            "number": 2,
            "instruction": "Rest 20 minutes, add salt, then mix until smooth.",
        },
        {
            "number": 3,
            "instruction": f"Ball into {config['ball_count']} pieces and lightly oil containers.",
        },
    ]

    if config["cold_ferment_hours"] > 0:
        steps.append(
            {
                "number": len(steps) + 1,
                "instruction": f"Cold ferment {config['cold_ferment_hours']} hours.",
            }
        )

    if config["room_temp_hours"] > 0:
        steps.append(
            {
                "number": len(steps) + 1,
                "instruction": f"Warm at room temp for {config['room_temp_hours']} hours before bake.",
            }
        )

    steps.append(
        {
            "number": len(steps) + 1,
            "instruction": f"Bake at {config['bake_temp_f']}F until blistered.",
        }
    )

    return steps


def _build_card(config: Dict[str, Any], ingredients_data: Dict[str, Any]) -> Dict[str, Any]:
    bake_temp_c = round((config["bake_temp_f"] - 32) * 5 / 9)
    title = f"{config['style'].title()} pizza dough"
    subtitle = (
        f"{config['ball_count']} x {int(config['ball_weight_g'])}g balls"
        f" | {config['hydration_percent']}% hydration"
    )
    return {
        "card_type": "recipe",
        "title": title,
        "subtitle": subtitle,
        "data": {
            "style": config["style"],
            "ball_count": config["ball_count"],
            "ball_weight_g": config["ball_weight_g"],
            "total_dough_g": ingredients_data["total_dough_g"],
            "hydration_percent": config["hydration_percent"],
            "salt_percent": config["salt_percent"],
            "yeast_percent": config["yeast_percent"],
            "oil_percent": config["oil_percent"],
            "bake_temp_f": config["bake_temp_f"],
            "bake_temp_c": bake_temp_c,
            "cold_ferment_hours": config["cold_ferment_hours"],
            "room_temp_hours": config["room_temp_hours"],
            "ingredients": ingredients_data["ingredients"],
            "steps": _build_steps(config),
            "updated_at": datetime.utcnow().isoformat(),
        },
    }


def pizza_dough_recipe(
    style: Optional[str] = None,
    ball_weight_g: Optional[float] = None,
    ball_count: Optional[int] = None,
    hydration_percent: Optional[float] = None,
    salt_percent: Optional[float] = None,
    yeast_percent: Optional[float] = None,
    oil_percent: Optional[float] = None,
    bake_temp_f: Optional[int] = None,
    cold_ferment_hours: Optional[float] = None,
    room_temp_hours: Optional[float] = None,
    use_last: Optional[bool] = True,
) -> str:
    """
    Build or update a pizza dough recipe using baker's percentages.

    If use_last is true, omitted parameters are filled from the most recent recipe.
    """
    global _LAST_RECIPE

    base = DEFAULT_CONFIG
    if use_last and _LAST_RECIPE:
        base = _LAST_RECIPE["config"]

    overrides = {
        "style": style or None,
        "ball_weight_g": _coerce_float(ball_weight_g, "ball_weight_g"),
        "ball_count": _coerce_int(ball_count, "ball_count"),
        "hydration_percent": _coerce_float(hydration_percent, "hydration_percent"),
        "salt_percent": _coerce_float(salt_percent, "salt_percent"),
        "yeast_percent": _coerce_float(yeast_percent, "yeast_percent"),
        "oil_percent": _coerce_float(oil_percent, "oil_percent"),
        "bake_temp_f": _coerce_int(bake_temp_f, "bake_temp_f"),
        "cold_ferment_hours": _coerce_float(cold_ferment_hours, "cold_ferment_hours"),
        "room_temp_hours": _coerce_float(room_temp_hours, "room_temp_hours"),
    }

    config = _merge_config(base, overrides)
    _validate_config(config)

    ingredients_data = _build_ingredients(config)
    card = _build_card(config, ingredients_data)

    _LAST_RECIPE = {"config": config, "card": card}
    _emit_ui_card(card)

    summary = (
        f"{card['title']}: {config['ball_count']} x {int(config['ball_weight_g'])}g"
        f" balls at {config['hydration_percent']}% hydration."
    )
    return summary
