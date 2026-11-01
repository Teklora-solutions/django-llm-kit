"""A small server-rendered column chart (SVG) for the admin dashboard: one series, one hue."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from django.utils.html import escape


def daily_columns(points: list[dict], days: int, *, end: date, fmt=lambda v: f"${v:.4f}") -> dict:
    """Fill missing days with 0 and lay out columns. Returns data for the template."""
    by_day = {}
    for p in points:
        b = p["bucket"]
        by_day[b.date() if isinstance(b, datetime) else b] = Decimal(p["value"] or 0)
    series = [
        (end - timedelta(days=days - 1 - i), by_day.get(end - timedelta(days=days - 1 - i), Decimal(0)))
        for i in range(days)
    ]
    peak = max((v for _, v in series), default=Decimal(0))
    width, height, top, bottom = 1040, 200, 22, 24  # drawn at about its displayed size, so 24px bars stay 24px
    plot_h = height - top - bottom
    slot = width / max(len(series), 1)
    bar_w = min(24.0, slot - 2)  # cap at 24px; the 2px is the surface gap
    peak_index = max(range(len(series)), key=lambda i: series[i][1]) if series else None
    bars = []
    for i, (day, value) in enumerate(series):
        h = 0 if peak == 0 else float(value / peak) * plot_h
        x = i * slot + (slot - bar_w) / 2
        y = top + plot_h - h
        label = fmt(value) if (i == peak_index and value > 0) or i == len(series) - 1 else ""
        bars.append(
            {
                "x": round(x, 1),
                "hit_x": round(i * slot, 1),
                "hit_w": round(slot, 1),
                "y": round(y, 1),
                "w": round(bar_w, 1),
                "h": round(h, 1),
                "path": _column(x, y, bar_w, h),
                "day": day,
                "title": escape(f"{day:%a %d %b}: {fmt(value)}"),
                "label": label,
                "label_x": round(x + bar_w / 2, 1),
                "label_y": round(y - 5, 1),
                "tick": day.strftime("%d %b") if (i % max(1, len(series) // 6) == 0) else "",
            }
        )
    return {
        "bars": bars,
        "width": width,
        "height": height,
        "baseline": top + plot_h,
        "tick_y": height - 6,
        "rows": series,
        "peak": peak,
    }


def _column(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Column path: 4px rounded data-end (top), square at the baseline."""
    if h <= 0:
        return ""
    r = min(r, w / 2, h)
    return (
        f"M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z"
    )
