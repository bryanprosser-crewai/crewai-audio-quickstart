"""Form schemas + session state machine — the voice-guided form-filling pattern.

The live Flow path is one LLM.call per field (JSON extract) plus canned
prompts and typed validation. tools.SetFieldTool / SubmitFormTool remain for
the unused Agent.kickoff form agent. Submission is mocked — a real
deployment POSTs to whatever records system the customer uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any

_VALID_TYPES = frozenset({"text", "number", "date", "choice"})


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    required: bool = True
    field_type: str = "text"  # text | number | date | choice
    options: tuple[str, ...] = ()
    hint: str = ""


@dataclass(frozen=True)
class FormSchema:
    form_type: str
    title: str
    fields: tuple[FormField, ...]


FORM_SCHEMAS: dict[str, FormSchema] = {
    "maintenance_report": FormSchema(
        form_type="maintenance_report",
        title="Maintenance Report",
        fields=(
            FormField("AssetId", "Asset ID", True, "text", hint="e.g. PUMP A1"),
            FormField("WorkDone", "Work Done", True, "text",
                      hint="Describe the maintenance performed."),
            FormField("TimeSpentHours", "Time Spent (hours)", True, "number",
                      hint="e.g. 1.5"),
            FormField("CompletionDate", "Completion Date", True, "date",
                      hint="e.g. 2026-07-13 or 'July 13 2026'"),
            FormField("ReportedBy", "Reported By", True, "text",
                      hint="Your name or badge ID."),
        ),
    ),
    "incident_report": FormSchema(
        form_type="incident_report",
        title="Incident Report",
        fields=(
            FormField("AssetId", "Asset ID", True, "text", hint="e.g. COMPRESSOR B1"),
            FormField("Severity", "Severity", True, "choice",
                      options=("Low", "Medium", "High"),
                      hint="Choose one: Low, Medium, High."),
            FormField("Description", "Description", True, "text",
                      hint="What happened?"),
            FormField("Notes", "Notes", False, "text",
                      hint="Anything else, or say 'none'."),
            FormField("ReportedBy", "Reported By", True, "text",
                      hint="Your name or badge ID."),
        ),
    ),
}


class FormSession:
    """Mutable state for one form-filling conversation."""

    def __init__(self, schema: FormSchema) -> None:
        self.schema = schema
        self.data: dict[str, Any] = {}
        self.submitted = False

    def state_summary(self) -> str:
        lines = [f"Form: {self.schema.title}"]
        for f in self.schema.fields:
            value = self.data.get(f.name, "(not set)")
            req = "" if f.required else " (optional)"
            lines.append(f"  {f.label}{req}: {value}")
        return "\n".join(lines)

    def missing_required(self) -> list[FormField]:
        return [f for f in self.schema.fields if f.required and f.name not in self.data]


    def next_open_field(self) -> FormField | None:
        """First schema field that does not yet have a stored value."""
        for f in self.schema.fields:
            if f.name not in self.data:
                return f
        return None


def ask_field(field: FormField) -> str:
    """Canned voice prompt for one field — no LLM."""
    extra = f" {field.hint}" if field.hint else ""
    if field.field_type == "choice" and field.options:
        extra = f" Choose one: {', '.join(field.options)}."
    return f"What's the {field.label}?{extra}"


def readback_and_confirm(session: FormSession) -> str:
    bits = [f"{f.label}: {session.data.get(f.name, '(blank)')}" for f in session.schema.fields]
    return ("I've got " + "; ".join(bits) + ". Say confirm to submit, or cancel to abort.")


def start_form_prompt(session: FormSession) -> str:
    first = session.schema.fields[0]
    return f"Starting the {session.schema.title}. {ask_field(first)}"


_EXTRACT_SYSTEM = """\
You extract one form-field value from a spoken or typed utterance.
Reply with JSON only, no markdown.
If you can read a value, return {"value": "<normalized string>"}.
Normalize numbers to digits (e.g. "five hours" → "5").
Normalize dates to YYYY-MM-DD when you can.
If the field is optional and the user declines (none, skip, no), return {"skip": true}.
If you cannot extract a usable value, return {"error": "unclear"}."""


def extract_system_prompt(field: FormField) -> str:
    opts = f" Options: {list(field.options)}." if field.options else ""
    return (
        f"{_EXTRACT_SYSTEM}\n"
        f"Field name: {field.name}\n"
        f"Label: {field.label}\n"
        f"Type: {field.field_type}.{opts}"
    )


def parse_extract_json(raw: str) -> dict:
    """Parse the slot-filler JSON; tolerate optional markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unclear"}
    return data if isinstance(data, dict) else {"error": "unclear"}


def validate_field(field: FormField, value: str) -> tuple[str, str | None]:
    """Normalise + validate a raw value; returns (value, error-or-None)."""
    if field.field_type == "choice":
        match = next((o for o in field.options if o.lower() == value.strip().lower()), None)
        if match is None:
            return value, (f"ERROR: '{value}' is not valid for {field.label}. "
                           f"Options: {field.options}")
        return match, None
    if field.field_type == "number":
        try:
            return str(float(value)), None
        except ValueError:
            return value, f"ERROR: '{value}' is not a number for {field.label}."
    if field.field_type == "date":
        for fmt in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(value.strip().replace(",", ""), fmt).strftime("%Y-%m-%d"), None
            except ValueError:
                continue
        return value, (f"ERROR: '{value}' is not a recognisable date for "
                       f"{field.label}. Try e.g. '2026-07-13'.")
    return value, None


def build_form_prompt(session: FormSession) -> str:
    """The form agent's marching orders, generated from the live schema."""
    fields = "\n".join(
        f"  - {f.name} ({f.label}) [{'Required' if f.required else 'Optional'}, "
        f"type={f.field_type}]"
        + (f" Options: {f.options}." if f.options else "")
        + (f" Hint: {f.hint}" if f.hint else "")
        for f in session.schema.fields
    )
    return (
        f"You are a voice-guided form assistant completing the "
        f"'{session.schema.title}' form.\n\nFORM FIELDS:\n{fields}\n\n"
        "RULES:\n"
        "1. Ask for ONE field at a time, starting with the first unfilled required field.\n"
        "2. Store each answer with set_field; if it returns an error, explain and re-ask.\n"
        "3. When all required fields are filled, read every value back and ask the user "
        "to say 'confirm' to submit or 'cancel' to abort.\n"
        "4. Only call submit_form after an explicit confirmation.\n"
        "5. Keep responses short — the user may be listening, not reading."
    )
