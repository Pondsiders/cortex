"""All of Cortex's time arithmetic and formatting.

Two rules live here rather than being spelled out at each call site. The day runs 6 AM
to 6 AM, so a memory stored at 2 AM belongs to the day before — grouping by calendar
date instead once misfiled 38 of 261 diary entries. And times shown to a reader are
local and American, never ISO and never 24-hour, because the two people reading them
are bad at timezone arithmetic and the format removes the chance to fumble it.

Times written to disk are the other way round: ISO-8601 with an offset, so they sort
and parse.
"""

from __future__ import annotations

from datetime import datetime

import pendulum

DAY_BOUNDARY_HOUR = 6
"""The day runs 6 AM to 6 AM, so a memory stored at 2 AM belongs to yesterday."""


def pondside_day(moment: datetime | None = None) -> str:
    """Return the ``YYYY-MM-DD`` day a moment falls in, on the 6 AM seam."""
    when = pendulum.instance(moment) if moment else pendulum.now()
    return when.subtract(hours=DAY_BOUNDARY_HOUR).format("YYYY-MM-DD")


def pso8601(moment: datetime) -> str:
    """Format a moment for a reader: ``Mon Aug 10 2026, 6:14 PM``."""
    return pendulum.instance(moment).format("ddd MMM D YYYY, h:mm A")


def age(moment: datetime) -> str:
    """Say how long ago a moment was, in words: ``2 hours ago``, ``5 weeks ago``."""
    return pendulum.instance(moment).diff_for_humans()


def gap(earlier: datetime, later: datetime | None = None) -> str:
    """Say how much time separates two moments: ``3 minutes``, ``2 hours``."""
    end = pendulum.instance(later) if later else pendulum.now()
    return end.diff_for_humans(pendulum.instance(earlier), absolute=True)


def gap_seconds(earlier: datetime, later: datetime | None = None) -> float:
    """Return the seconds between two moments, for a record rather than a reader."""
    end = pendulum.instance(later) if later else pendulum.now()
    return (end - pendulum.instance(earlier)).total_seconds()
