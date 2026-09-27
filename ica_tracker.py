#!/usr/bin/env python3
"""
Singapore ICA 60/90-day rolling rule tracker for Malaysian STVP holders.

Physical presence: entry and exit calendar days both count (inclusive).
Trailing window at evaluation date T: [T - 89 days, T] (90 calendar days).
Maximum allowed presence in that window: 60 days.

This is NOT "must stay outside Singapore for 30 days." Days abroad do not add to
the 60; old in-SG days keep counting until they fall off the back of the 90-day
window. There is no fixed "reset" period—use plan-outside to see how long to
stay out before enough allowance returns.

Dates are plain calendar dates (local date.today()); no timezone conversion.

Travel data layout (recommended):
  travel_history/2026.csv, travel_history/2027.csv, ...
One row per trip. Dates in CSV use DD/MM/YYYY. Trips that cross 31 Dec -> 1 Jan
stay on a single row; store that row in the file for the entry year (e.g. 2026.csv).
All year files are merged when calculating the rolling window.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

MAX_DAYS_IN_WINDOW = 60
WINDOW_LENGTH_DAYS = 90
DATE_FMT = "%d/%m/%Y"
CSV_HEADERS = ("user", "entry_date", "exit_date", "notes")
DEFAULT_DATA_DIR = Path("travel_history")
YEAR_CSV_PATTERN = re.compile(r"^(\d{4})\.csv$")
SAFE_RETURN_SEARCH_LIMIT_DAYS = 400


@dataclass(frozen=True)
class Trip:
    entry_date: date
    exit_date: Optional[date]  # None = ongoing through evaluation date
    notes: str = ""


@dataclass
class SimResult:
    approved: bool
    first_fail_day: Optional[date]
    max_used_during_stay: int
    stay_end: date


def parse_date(value: str, context: str) -> date:
    value = (value or "").strip()
    try:
        return datetime.strptime(value, DATE_FMT).date()
    except ValueError as exc:
        raise ValueError(f"{context}: invalid date '{value}' (expected DD/MM/YYYY)") from exc


def ensure_year_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)


def ensure_data_dir(data_dir: Path, year: Optional[int] = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    y = year if year is not None else date.today().year
    ensure_year_csv(data_dir / f"{y}.csv")


def list_year_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    files: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        match = YEAR_CSV_PATTERN.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def warn_cross_year_file_mismatch(path: Path, file_year: int, trip: Trip) -> None:
    exit_eff = trip.exit_date or trip.entry_date
    years = {trip.entry_date.year, exit_eff.year}
    if file_year not in years and len(years) > 1:
        print(
            f"Warning: {path.name} is for calendar year {file_year} but trip "
            f"{format_date(trip.entry_date)} -> "
            f"{format_date(trip.exit_date) if trip.exit_date else '(open)'} "
            f"spans other years; keep cross-year trips in the entry-year file.",
            file=sys.stderr,
        )
    elif trip.entry_date.year != file_year:
        print(
            f"Warning: {path.name} is for {file_year} but trip entry is "
            f"{format_date(trip.entry_date)}; prefer storing in "
            f"{trip.entry_date.year}.csv.",
            file=sys.stderr,
        )


def effective_exit(trip: Trip, evaluation: date) -> date:
    if trip.exit_date is None:
        return evaluation
    return trip.exit_date


def trips_overlap(a: Trip, b: Trip) -> bool:
    """True if two trips could double-count the same calendar day (open ends treated as far future)."""
    far = date.max
    a_end = a.exit_date or far
    b_end = b.exit_date or far
    return a.entry_date <= b_end and b.entry_date <= a_end


def check_overlaps(trips: list[Trip]) -> list[tuple[int, int]]:
    overlaps: list[tuple[int, int]] = []
    for i in range(len(trips)):
        for j in range(i + 1, len(trips)):
            if trips_overlap(trips[i], trips[j]):
                overlaps.append((i, j))
    return overlaps


def _parse_trip_row(
    path: Path,
    line_no: int,
    row: dict[str, str],
    user: str,
    file_year: Optional[int],
) -> Optional[Trip]:
    row_user = (row.get("user") or "").strip()
    if row_user != user:
        return None
    entry_raw = (row.get("entry_date") or "").strip()
    exit_raw = (row.get("exit_date") or "").strip()
    notes = (row.get("notes") or "").strip()
    if not entry_raw:
        raise ValueError(f"{path} line {line_no}: missing entry_date for user '{user}'")
    entry = parse_date(entry_raw, f"{path} line {line_no} entry_date")
    exit_d: Optional[date] = None
    if exit_raw:
        exit_d = parse_date(exit_raw, f"{path} line {line_no} exit_date")
        if exit_d < entry:
            raise ValueError(f"{path} line {line_no}: exit_date before entry_date")
    trip = Trip(entry_date=entry, exit_date=exit_d, notes=notes)
    if file_year is not None:
        warn_cross_year_file_mismatch(path, file_year, trip)
    return trip


def load_trips_from_file(path: Path, user: str, file_year: Optional[int] = None) -> list[Trip]:
    user = user.strip()
    trips: list[Trip] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return trips
        missing = set(CSV_HEADERS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path}: missing CSV columns {sorted(missing)}; expected {list(CSV_HEADERS)}"
            )
        for line_no, row in enumerate(reader, start=2):
            trip = _parse_trip_row(path, line_no, row, user, file_year)
            if trip is not None:
                trips.append(trip)
    return trips


def load_trips(path: Path, user: str) -> list[Trip]:
    """
    Load trips from a single CSV file or merge all YYYY.csv files under a directory.
    """
    user = user.strip()
    if path.is_dir():
        ensure_data_dir(path)
        year_files = list_year_csv_files(path)
        if not year_files:
            ensure_data_dir(path, date.today().year)
            year_files = list_year_csv_files(path)
        trips: list[Trip] = []
        for year_path in year_files:
            year = int(YEAR_CSV_PATTERN.match(year_path.name).group(1))
            trips.extend(load_trips_from_file(year_path, user, file_year=year))
        trips.sort(key=lambda t: t.entry_date)
        return trips

    if path.is_file():
        return sorted(load_trips_from_file(path, user), key=lambda t: t.entry_date)

    ensure_year_csv(path)
    return load_trips_from_file(path, user)


def presence_day_set(trips: Iterable[Trip], evaluation: date) -> set[date]:
    window_start = evaluation - timedelta(days=WINDOW_LENGTH_DAYS - 1)
    days: set[date] = set()
    for trip in trips:
        exit_eff = effective_exit(trip, evaluation)
        start = max(trip.entry_date, window_start)
        end = min(exit_eff, evaluation)
        if start > end:
            continue
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return days


def days_used(trips: list[Trip], evaluation: date) -> int:
    return len(presence_day_set(trips, evaluation))


def remaining_allowance(used: int) -> int:
    return max(0, MAX_DAYS_IN_WINDOW - used)


def simulate_stay(trips: list[Trip], entry: date, stay_days: int) -> SimResult:
    if stay_days < 1:
        raise ValueError("stay duration must be at least 1 day")
    stay_end = entry + timedelta(days=stay_days - 1)
    hypothetical = Trip(entry_date=entry, exit_date=stay_end, notes="(simulated)")
    combined = trips + [hypothetical]
    first_fail: Optional[date] = None
    max_used = 0
    d = entry
    while d <= stay_end:
        used = days_used(combined, d)
        max_used = max(max_used, used)
        if used > MAX_DAYS_IN_WINDOW and first_fail is None:
            first_fail = d
        d += timedelta(days=1)
    return SimResult(
        approved=first_fail is None,
        first_fail_day=first_fail,
        max_used_during_stay=max_used,
        stay_end=stay_end,
    )


def find_safe_return_date(
    trips: list[Trip],
    start: date,
    target_remaining: int = 30,
) -> Optional[date]:
    """Earliest date F >= start where remaining allowance is at least target_remaining."""
    if target_remaining < 1 or target_remaining > MAX_DAYS_IN_WINDOW:
        raise ValueError(f"target_remaining must be 1..{MAX_DAYS_IN_WINDOW}")
    target_used = MAX_DAYS_IN_WINDOW - target_remaining
    if days_used(trips, start) <= target_used:
        return start
    for offset in range(1, SAFE_RETURN_SEARCH_LIMIT_DAYS + 1):
        candidate = start + timedelta(days=offset)
        if days_used(trips, candidate) <= target_used:
            return candidate
    return None


def close_trips_at_exit(trips: list[Trip], exit_date: date) -> list[Trip]:
    """For planning: end open visits on exit_date; drop future-dated entries."""
    closed: list[Trip] = []
    for trip in trips:
        if trip.entry_date > exit_date:
            continue
        if trip.exit_date is None:
            closed.append(Trip(trip.entry_date, exit_date, trip.notes))
        else:
            closed.append(trip)
    return closed


def days_outside_between(exit_date: date, return_date: date) -> int:
    """
    Full calendar days spent outside SG between visits.
    exit_date and return_date are both in-SG days (last day in / first day back).
    """
    if return_date <= exit_date:
        return 0
    return (return_date - exit_date).days - 1


def trip_days_contributing_in_window(trip: Trip, evaluation: date) -> int:
    exit_eff = effective_exit(trip, evaluation)
    window_start = evaluation - timedelta(days=WINDOW_LENGTH_DAYS - 1)
    start = max(trip.entry_date, window_start)
    end = min(exit_eff, evaluation)
    if start > end:
        return 0
    return (end - start).days + 1


def find_earliest_return_after_exit(
    trips: list[Trip],
    exit_date: date,
    target_remaining: int = 30,
) -> Optional[tuple[date, int]]:
    """
    After last in-SG day exit_date, earliest return_date (entry day) with enough allowance.
    Returns (return_date, days_outside_before_return).
    """
    closed = close_trips_at_exit(trips, exit_date)
    first_day_out = exit_date + timedelta(days=1)
    found = find_safe_return_date(closed, first_day_out, target_remaining=target_remaining)
    if found is None:
        return None
    if found <= exit_date:
        return_date = exit_date + timedelta(days=1)
    else:
        return_date = found
    return return_date, days_outside_between(exit_date, return_date)


def format_date(d: date) -> str:
    return d.strftime(DATE_FMT)


def warn_overlaps(trips: list[Trip], user: str) -> None:
    overlaps = check_overlaps(trips)
    for i, j in overlaps:
        print(
            f"Warning: overlapping trips for user '{user}' "
            f"(entries {format_date(trips[i].entry_date)} and "
            f"{format_date(trips[j].entry_date)})",
            file=sys.stderr,
        )


def cmd_summary(trips: list[Trip], user: str, evaluation: date, breakdown: bool) -> int:
    warn_overlaps(trips, user)
    used = days_used(trips, evaluation)
    remaining = remaining_allowance(used)
    window_start = evaluation - timedelta(days=WINDOW_LENGTH_DAYS - 1)
    over = used > MAX_DAYS_IN_WINDOW

    print(f"User: {user}")
    print(f"Evaluation date: {format_date(evaluation)}")
    print(f"90-day window: {format_date(window_start)} to {format_date(evaluation)}")
    print(f"Days used in window: {used} / {MAX_DAYS_IN_WINDOW}")
    print(f"Remaining allowance: {remaining}")
    print(f"Over limit: {'yes' if over else 'no'}")

    if breakdown and trips:
        print("\nDays in window by trip (only physical days in SG count):")
        for trip in trips:
            n = trip_days_contributing_in_window(trip, evaluation)
            exit_s = format_date(trip.exit_date) if trip.exit_date else "(open)"
            print(
                f"  {format_date(trip.entry_date)} -> {exit_s}: {n} day(s)  {trip.notes or ''}".rstrip()
            )

    if trips:
        print("\nTrips (for this user):")
        for t in trips:
            exit_s = format_date(t.exit_date) if t.exit_date else "(open)"
            print(f"  {format_date(t.entry_date)} -> {exit_s}  {t.notes or ''}".rstrip())
    else:
        print("\nNo trips recorded for this user.")
    return 0


def cmd_simulate(
    trips: list[Trip], user: str, entry: date, stay_days: int, evaluation: date
) -> int:
    warn_overlaps(trips, user)
    result = simulate_stay(trips, entry, stay_days)
    used_at_entry = days_used(trips, entry)

    print(f"User: {user}")
    print(f"Planned entry: {format_date(entry)}")
    print(f"Planned stay: {stay_days} day(s) (through {format_date(result.stay_end)})")
    print(f"Days already used in window on entry date: {used_at_entry}")
    print(f"Peak days used during planned stay: {result.max_used_during_stay}")
    if result.approved:
        print("ICA simulation: PASS (stay would not exceed 60 days in any rolling 90-day window)")
    else:
        print(
            f"ICA simulation: FAIL (limit exceeded on {format_date(result.first_fail_day)}; "
            "likely auto-rejection)"
        )
    return 0 if result.approved else 1


def cmd_safe_return(trips: list[Trip], user: str, evaluation: date, target: int) -> int:
    warn_overlaps(trips, user)
    used = days_used(trips, evaluation)
    remaining = remaining_allowance(used)

    print(f"User: {user}")
    print(f"Evaluation date: {format_date(evaluation)}")
    print(f"Remaining allowance: {remaining}")
    print(f"Planning target: {target} day(s) remaining in rolling window")

    if remaining >= target:
        print(
            f"No wait needed: you already have at least {target} days remaining "
            f"in the rolling 90-day window."
        )
        return 0

    found = find_safe_return_date(trips, evaluation, target_remaining=target)
    if found is None:
        print(
            f"Could not find a date within {SAFE_RETURN_SEARCH_LIMIT_DAYS} days "
            f"with {target} days remaining. Review travel history or extend search."
        )
        return 1

    used_then = days_used(trips, found)
    rem_then = remaining_allowance(used_then)
    if found == evaluation:
        print(f"Earliest date with {target}+ days remaining: {format_date(found)}")
    else:
        print(
            f"Earliest future date with at least {target} days remaining: "
            f"{format_date(found)} (remaining then: {rem_then})"
        )
    return 0


def cmd_plan_outside(
    trips: list[Trip], user: str, exit_date: date, target: int
) -> int:
    warn_overlaps(trips, user)
    closed = close_trips_at_exit(trips, exit_date)
    used_on_exit = days_used(closed, exit_date)

    print(f"User: {user}")
    print(f"Last day in Singapore (exit): {format_date(exit_date)}")
    print(f"Days used in 90-day window on that exit day: {used_on_exit}")
    print(f"Goal: at least {target} day(s) allowance when they re-enter")
    print(
        "\nNote: 60/90 is rolling presence in SG, not a fixed '30 days outside' rule."
    )

    result = find_earliest_return_after_exit(closed, exit_date, target_remaining=target)
    if result is None:
        print(
            f"\nNo suitable re-entry found within {SAFE_RETURN_SEARCH_LIMIT_DAYS} days. "
            "Try a lower target or check history."
        )
        return 1

    return_date, outside_days = result
    used_on_return = days_used(closed, return_date)
    rem = remaining_allowance(used_on_return)

    print(f"\nEarliest re-entry date (first day back in SG): {format_date(return_date)}")
    print(f"Full days outside Singapore before that: {outside_days}")
    if outside_days == 0:
        print("(Re-entry can be the day after exit if allowance is already enough.)")
    else:
        first_out = exit_date + timedelta(days=1)
        last_out = return_date - timedelta(days=1)
        print(
            f"Outside period: {format_date(first_out)} to {format_date(last_out)} "
            f"({outside_days} day(s))"
        )
    print(f"Allowance on re-entry day: {rem} / {MAX_DAYS_IN_WINDOW} (used {used_on_return})")

    if target == MAX_DAYS_IN_WINDOW:
        print("This is a full 'clean' window (no prior SG days in the last 90 days).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track Singapore ICA 60/90-day rolling presence for STVP holders.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="User identifier matching the 'user' column in the CSV",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder with per-year CSV files (2026.csv, 2027.csv, ...) (default: travel_history)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional single CSV file instead of --data-dir (legacy)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Evaluation anchor date DD/MM/YYYY (default: today)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    summ = sub.add_parser("summary", help="Show current 90-day window usage and remaining allowance")
    summ.add_argument(
        "--breakdown",
        action="store_true",
        help="Show how many window days each trip contributes",
    )

    sim = sub.add_parser("simulate", help="Test a planned entry and stay duration")
    sim.add_argument("--entry", required=True, help="Planned entry date DD/MM/YYYY")
    sim.add_argument(
        "--days",
        type=int,
        required=True,
        help="Planned length of stay in days (entry day counts as day 1)",
    )

    safe = sub.add_parser(
        "safe-return",
        help="Find when enough rolling allowance is available again (from today)",
    )
    safe.add_argument(
        "--target",
        type=int,
        default=30,
        help="Desired remaining days in window (default: 30; use 60 for fully clear)",
    )

    plan = sub.add_parser(
        "plan-outside",
        help="After an exit, how long to stay out before re-entry with enough allowance",
    )
    plan.add_argument(
        "--exit",
        required=True,
        help="Last day in Singapore before the break (DD/MM/YYYY)",
    )
    plan.add_argument(
        "--target",
        type=int,
        default=30,
        help="Desired remaining days on re-entry (default: 30; use 60 for full reset)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    evaluation = date.today()
    if args.date:
        evaluation = parse_date(args.date, "--date")

    data_path = args.csv if args.csv is not None else args.data_dir

    try:
        trips = load_trips(data_path, args.user)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.command == "summary":
        return cmd_summary(trips, args.user, evaluation, breakdown=args.breakdown)
    if args.command == "simulate":
        try:
            entry = parse_date(args.entry, "simulate --entry")
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if args.days < 1:
            print("Error: --days must be a positive integer", file=sys.stderr)
            return 2
        return cmd_simulate(trips, args.user, entry, args.days, evaluation)
    if args.command == "safe-return":
        if args.target < 1 or args.target > MAX_DAYS_IN_WINDOW:
            print("Error: --target must be between 1 and 60", file=sys.stderr)
            return 2
        return cmd_safe_return(trips, args.user, evaluation, target=args.target)

    if args.command == "plan-outside":
        if args.target < 1 or args.target > MAX_DAYS_IN_WINDOW:
            print("Error: --target must be between 1 and 60", file=sys.stderr)
            return 2
        try:
            exit_d = parse_date(args.exit, "plan-outside --exit")
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        return cmd_plan_outside(trips, args.user, exit_d, target=args.target)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
