import io
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

