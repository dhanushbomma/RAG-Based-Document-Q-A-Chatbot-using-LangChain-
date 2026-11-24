import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from reportlab.pdfgen import canvas


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "handbook.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "Employees receive 25 days of paid vacation per year.")
    c.showPage()
    c.drawString(72, 750, "Remote work requires manager approval.")
    c.save()
    return path
