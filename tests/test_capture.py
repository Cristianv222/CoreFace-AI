"""
Tests para captura de rostros
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.capture import CapturadorInteligente

def test_capturador():
    capturador = CapturadorInteligente()
    assert capturador is not None