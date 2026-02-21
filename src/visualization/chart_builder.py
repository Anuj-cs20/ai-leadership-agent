"""Plotly chart generation and report formatting for leadership insights."""

import logging
from typing import Optional

import plotly.graph_objects as go

from src.generation.output_schema import ChartData, QueryResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _line_chart(cd: ChartData) -> go.Figure:
    labels = cd.data.get("labels", cd.data.get("x", []))
    values = cd.data.get("values", cd.data.get("y", []))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values,
        mode="lines+markers",
        name=cd.title,
        line=dict(color="#4A90D9", width=3),
        marker=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=cd.title, font=dict(size=16)),
        xaxis_title=cd.data.get("x_label", ""),
        yaxis_title=cd.data.get("y_label", ""),
        template="plotly_white",
        height=420, width=700,
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return fig


def _bar_chart(cd: ChartData) -> go.Figure:
    labels = cd.data.get("labels", cd.data.get("x", []))
    values = cd.data.get("values", cd.data.get("y", []))

    avg_val = sum(values) / len(values) if values else 0
    colors = ["#2ECC71" if v >= avg_val else "#E74C3C" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=cd.title, font=dict(size=16)),
        xaxis_title=cd.data.get("x_label", ""),
        yaxis_title=cd.data.get("y_label", ""),
        template="plotly_white",
        height=420, width=700,
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return fig


def _pie_chart(cd: ChartData) -> go.Figure:
    labels = cd.data.get("labels", [])
    values = cd.data.get("values", [])

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels, values=values, hole=0.3,
        marker_colors=["#4A90D9", "#2ECC71", "#E74C3C", "#E67E22", "#9B59B6", "#1ABC9C"],
    ))
    fig.update_layout(
        title=dict(text=cd.title, font=dict(size=16)),
        template="plotly_white",
        height=420, width=700,
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chart(response: QueryResponse) -> Optional[go.Figure]:
    """Build a Plotly chart from the QueryResponse visualization data.

    Returns None if no chart data is present.
    """
    if not response.visualization:
        return None

    cd = response.visualization
    chart_type = cd.chart_type.lower()

    try:
        if chart_type == "line":
            return _line_chart(cd)
        elif chart_type == "bar":
            return _bar_chart(cd)
        elif chart_type == "pie":
            return _pie_chart(cd)
        else:
            logger.warning(f"Unknown chart type '{chart_type}', defaulting to bar")
            return _bar_chart(cd)
    except Exception as e:
        logger.error(f"Chart build failed: {e}")
        return None


def format_report(response: QueryResponse) -> str:
    """Format a QueryResponse as a readable text report."""
    lines = [
        "📊 LEADERSHIP INSIGHT REPORT",
        "═" * 50,
        "",
        f'Question: "{response.question}"',
        f"Query Type: {response.query_type}",
        "",
        "📝 Summary",
        response.answer.summary,
        "",
        "🔑 Key Points",
    ]

    for point in response.answer.key_points:
        lines.append(f"  • {point}")
    lines.append("")

    if response.answer.evidence:
        lines.append("📄 Evidence")
        for ev in response.answer.evidence:
            lines.append(f'  "{ev.quote}"')
            src = f"    — {ev.source_document}"
            if ev.page_number:
                src += f", p.{ev.page_number}"
            if ev.section:
                src += f", §{ev.section}"
            lines.append(src)
            lines.append("")

    if response.answer.detailed_analysis:
        lines.append("📋 Detailed Analysis")
        lines.append(response.answer.detailed_analysis)
        lines.append("")

    if response.sources:
        lines.append("📎 Sources")
        for i, s in enumerate(response.sources, 1):
            entry = f"  {i}. {s.document_name}"
            if s.section_title:
                entry += f" → {s.section_title}"
            if s.page_number:
                entry += f" (p.{s.page_number})"
            lines.append(entry)
        lines.append("")

    if response.visualization:
        lines.append(
            f"📈 [{response.visualization.chart_type.title()} Chart: "
            f"{response.visualization.title}]"
        )

    return "\n".join(lines)
