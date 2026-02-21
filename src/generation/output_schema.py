"""Pydantic output models for structured leadership responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A direct quote from a source document supporting the answer."""

    quote: str = Field(description="Direct quote from the source document")
    source_document: str = Field(description="Filename of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number if available")
    section: Optional[str] = Field(default=None, description="Section header if available")


class Source(BaseModel):
    """A source document referenced in the answer."""

    document_name: str = Field(description="Filename of the source document")
    document_type: Optional[str] = Field(default=None, description="Document type: annual, quarterly, strategy, operational")
    page_number: Optional[int] = Field(default=None, description="Page number if available")
    section_title: Optional[str] = Field(default=None, description="Section title if available")


class ChartData(BaseModel):
    """Data for auto-generated Plotly charts."""

    chart_type: str = Field(description="Chart type: line, bar, pie")
    title: str = Field(description="Chart title")
    data: dict = Field(description="Chart data with 'labels' and 'values' keys, optionally 'x_label' and 'y_label'")
    description: str = Field(description="Brief description of what the chart shows")


class Answer(BaseModel):
    """Structured answer with summary, key points, and evidence."""

    summary: str = Field(description="Executive summary in 2-3 sentences")
    key_points: List[str] = Field(description="3-5 key findings as bullet points")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence quotes from documents")
    detailed_analysis: Optional[str] = Field(default=None, description="Optional deeper analysis")


class QueryResponse(BaseModel):
    """Complete structured response for a leadership query."""

    question: str = Field(description="The original user question")
    query_type: str = Field(description="One of: trend, comparison, risk, general")
    answer: Answer
    sources: List[Source] = Field(default_factory=list)
    visualization: Optional[ChartData] = Field(
        default=None,
        description="Chart data for trend/comparison queries. Include for trend (line chart) and comparison (bar chart) queries.",
    )
