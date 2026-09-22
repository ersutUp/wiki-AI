import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Literal

class OcrFileType(StrEnum):
    PDF = "pdf"
    Image = "image"
    Word = "word"
    Excel = "excel"
    PPT = "ppt"

@dataclass
class OcrDocument:
    class F:
        FILE_PATH  = "file_path"
        FILE_TYPE  = "file_type"
        PAGE_NO    = "pageNo"
        SOURCE     = "source"
        DOC_TYPE   = "doc_type"
        CHAPTER    = "chapter"
        CONTENT    = "content"
        EMBEDDINGS = "embeddings"
        SPARSE     = "sparse_embeddings"

    file_path:  Optional[str] = None
    doc_type:   Optional[str] = None
    content:    Optional[str] = None
    embeddings: Optional[list[float]] = None
    file_type:  OcrFileType = OcrFileType.PDF
    pageNo:     Optional[str] = None
    source:     Optional[str] = None
    chapter:    Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)