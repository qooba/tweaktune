import io
from enum import Enum
import pyarrow as pa
import pyarrow.ipc as ipc

def record_batches_to_ipc_bytes(reader: pa.RecordBatchReader) -> bytes:
    """
    Converts a RecordBatchReader to bytes using IPC format.
    """
    sink = io.BytesIO()
    writer = ipc.new_stream(sink, reader.schema)
    for batch in reader:
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()

def package_installation_hint(package_name: str):
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = "\033[1m"
    print(f"\t{BOLD}Please install:{ENDC}\t{OKGREEN}{package_name}{ENDC}{BOLD}{ENDC}")

class StepStatus(Enum):
    """Enum for step status."""
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

    def __str__(self):
        return self.value

class LogLevel(Enum):
    """Enum for log levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

    def __str__(self):
        return self.value


class DebugTargets(Enum):
    """Enum for debug targets."""
    EXTRACT_JSON = "extract_json"
    MISTRAL_LLM = "mistral_llm"
    UNSLOTH_LLM = "unsloth_llm"
    IFELSE_STEP = "ifelsestep"
    PYSTEP = "pystep"
    VALIDATE_JSON_STEP = "validate_json_step"
    TEXT_GENERATION_STEP = "text_generation_step"
    JSON_GENERATION_STEP = "json_generation_step"
    JSON_WRITER_STEP = "json_writer_step"
    TEMPLATES = "templates"
    TEMPLATES_ERR = "templates_err"

    def __str__(self):
        return self.value